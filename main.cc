  // === C++ Includes ===
#include <iostream>
#include <algorithm>    // std::sort

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

  // === my includes ===
#include <surface_communicator.h>
#include <refinement_communicator.h>
#include <maxwell_solver.h>

// Grid
#include <kirasfm_grid_generator.h>
#include <deal.II/grid/grid_out.h>

namespace KirasFM {
  using namespace dealii;

  // Switch to selcet between full sphere , half sphere or quarter_sphere
  //unsigned int selector = 0; // full sphere
  unsigned int selector = 1; // half sphere
  //unsigned int selector = 2; // eighth sphere

  template<int dim>
  class DDM {
    public:
      //constructor:
      DDM(
        boost::mpi::communicator world,
        ParameterReader &prm,
        ConditionalOStream pcout,
        TimerOutput timer,
        const unsigned int cpus_per_domain,
        const unsigned int slizes
      );

      void initialize();
      void step(std::vector<std::vector<unsigned int>> connectivity);

      // refinement methods:
      void prepare_refine(std::vector<std::vector<unsigned int>> connectivity);
      void refine();
      void mark_circular(double radius);
      void mark_circular_coarser(double radius);
      void mark_shell_coarser(double inner_radius, double outer_radius);
      void mark_adaptive();

      void fix_stupid_material_id();

      void print_result() const;

    private:

      // MPI
      boost::mpi::communicator world;
      const unsigned int world_size;
      const unsigned int world_rank;

      // Parameters
      ParameterReader prm;

      std::vector<unsigned int> proc_id_list; 
      const unsigned int cpus_per_domain;

      // information about the system
      const unsigned int slizes;
      const unsigned int size;

      // MPI communicatior
      MPI_Comm mpi_local_comm;
      std::vector<unsigned int> owned_problems;
      std::vector<std::vector<unsigned int>> domain_map;

      // the system:
      std::vector<MaxwellProblem<dim>> thm;
      SurfaceCommunicator<dim> g_out, g_in;

      RefinementCommunicator<dim> r_in;

      // help functions
      bool proc_first() const;
      unsigned int proc_previous() const;
      unsigned int proc_size() const;
  };

  template<int dim>
  DDM<dim>::DDM (
    boost::mpi::communicator world,
    ParameterReader &prm,
    ConditionalOStream pcout,
    TimerOutput timer,
    const unsigned int cpus_per_domain,
    const unsigned int slizes
  ) :

    world(world),
    world_size( Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) ),
    world_rank( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ),

    prm(prm), 

    cpus_per_domain(cpus_per_domain),

    slizes(slizes),
    size( (selector == 0) ? slizes : ( (selector == 1) ? slizes * 2 : slizes * 8) ),

    domain_map(std::vector<std::vector<unsigned int>>(size)),

    // Initialize the SurfaceCommunicator
    g_out( SurfaceCommunicator<dim>(size) ),
    g_in( SurfaceCommunicator<dim>(size) ),

    // Initialize the RefinementCommunicator, with size domains
    r_in( RefinementCommunicator<dim>(size) )

  {
    if(world_size < cpus_per_domain) {
      pcout << "WARNING: You tried to assaign to each domain " << cpus_per_domain 
            << " threads, but there are only " << world_size << "threads available!" 
            << std::endl;
    }

    if(world_size > size * cpus_per_domain ) {
      pcout << "WARNING: There are " << size << " tasks to solve, but there are "
            << world_size << " threads available, therefore " << world_size - size 
            << " remain unused!" << std::endl;
    }

    // distribute the indicies fot the CPUs
    for( unsigned int proc_id = 0; proc_id * cpus_per_domain < world_size; proc_id++ ){
      for( unsigned int i = 0; i < cpus_per_domain; i++) {
        proc_id_list.push_back( proc_id );
      }
    }

    // assign each CPU to a certain domain
    MPI_Comm_split(MPI_COMM_WORLD, proc_id_list[world_rank], world_rank, &mpi_local_comm);

    // number of domains that can be evaluated in parallel:
    unsigned int n_chunks = world_size / cpus_per_domain;
    if( world_size % cpus_per_domain != 0 )
      n_chunks++;

    // number of domains each chunk has to solve
    for(unsigned int i = 0; i < size; i++) {
      if( i % n_chunks ==  proc_id_list[world_rank] ) {
        unsigned int pos = (i / n_chunks) * n_chunks; // since i and n_chunks are integers, (i / n_chunks) is adjusted downards.
        owned_problems.push_back(pos + proc_id_list[world_rank]);
      }
    }

    // create the domain map (a vector that contains on position domain_id the process_id that owns this domain)
    std::vector<std::vector<unsigned int>> gathered_problems = Utilities::MPI::all_gather(MPI_COMM_WORLD, owned_problems);
    for(unsigned int i = 0; i < world_size; i++) {
      for(unsigned int j : gathered_problems[i]) 
        domain_map[j].push_back(i);
    }

    // create list of Maxwell Problems:
    for( unsigned int i: owned_problems)  
      thm.push_back( MaxwellProblem<dim>(prm, pcout, timer, i, size , mpi_local_comm) );
    
  }

  template<int dim>
  void DDM<dim>::initialize() {
    //if( first_id(proc_id_list, world_rank) ) {

    // set up grid:
    const unsigned int refinements = prm.get_integer("Mesh & geometry parameters", "Number of refinements");
//    const unsigned int scale       = prm.get_double("Mesh & geometry parameters", "Size of grid");

    for( unsigned int i = 0; i < owned_problems.size(); i++ ) {
      KirasFM_Grid_Generator::KirasFMGridGenerator<dim> ddm_gg(owned_problems[i], size, refinements);
      //  Silver ball in vacuum (3D only)

      std::vector<double> layer_thickness = {2.0, 1.7, 0.95, 0.7};
      if ( selector == 0 ) {
        ddm_gg.create_nano_particle(
          thm[i].return_triangulation(),
          1.0, /*  radius of the silver ball */
          layer_thickness
        );
      }
      else if ( selector == 1 ) {
        ddm_gg.create_half_nano_particle(
          thm[i].return_triangulation(),
          1.0, /*  radius of the silver ball */
          layer_thickness
        );
      }
      else if ( selector == 2 ) {
        ddm_gg.create_eighth_nano_particle(
          thm[i].return_triangulation(),
          1.0, /*  radius of the silver ball */
          layer_thickness
        );
      }
      else {
        Assert(false, ExcInternalError());
      }

      mark_circular_coarser(0.87);
      mark_shell_coarser(1.05, 3.0);
      //prepare_refine(connectivity);
      refine();
      mark_circular_coarser(0.92);
      mark_shell_coarser(1.4, 3.0);
      //problem.prepare_refine(connectivity);
      refine();

      fix_stupid_material_id();

      //std::string name = "Grid-" + std::to_string(owned_problems[i]) + ".vtk";
      //std::ofstream output_file1(name.c_str());
      //GridOut().write_vtk(thm[i].return_triangulation(), output_file1);
    }

    // initalize the maxwell problems:
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].initialize();

    // we already solve it the first time here:
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].solve();

  }

  template<int dim>
  void DDM<dim>::step(std::vector<std::vector<unsigned int>> connectivity) {

    // update the interfaces (compute g_in)
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].update_interface_rhs();

    // Gather g_in:
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      g_in.update(thm[i].return_g_in(), owned_problems[i]);

    // distribute g_in to all neighbours:
    if( proc_first() ) {
      for(unsigned int problem : owned_problems) {
        for(unsigned int dest : connectivity[problem]) {
          const unsigned int i = domain_map[dest][0];
          SurfaceCommunicator<dim> g_tmp(size);

          // send & recieve
          boost::mpi::request reqs[2];
          reqs[0] = world.isend(i, dest, g_in);
          reqs[1] = world.irecv(i, problem, g_tmp);
          boost::mpi::wait_all(reqs, reqs + 2 );
          g_in.update(g_tmp, dest);
        }
      }

      for(unsigned int i = 1; i < proc_size(); i++) 
        world.send(world_rank + i, world_rank + i, g_in);

    } else {
      SurfaceCommunicator<dim> g_tmp(size);
      world.recv(proc_previous(), world_rank, g_tmp);
      g_in = g_tmp;
    }


    // Distribute g_in:
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].update_g_in(g_in);

    // compute with the new g_in the interface right hand side (and compute g_out)
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].assemble_interface_rhs();

    //// Gather g_out
    //for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
    //  g_out.update(thm[i].return_g_out(), owned_problems[i]);
    //
    //if( proc_first() ) {
    //  for(unsigned int problem : owned_problems) {
    //    for(unsigned int dest : connectivity[problem]) {
    //      const unsigned int i = domain_map[dest][0];
    //      SurfaceCommunicator<dim> g_tmp(size);
    //
    //      // send & recieve
    //      boost::mpi::request reqs[2];
    //      reqs[0] = world.isend(i, i, g_out);
    //      reqs[1] = world.irecv(i, world_rank, g_tmp);
    //      boost::mpi::wait_all(reqs, reqs + 2 );
    //      g_out.update(g_tmp, dest);
    //    }
    //  }
    //
    //  for(unsigned int i = 1; i < proc_size(); i++)
    //    world.send(world_rank + i, world_rank + i, g_out);
    //
    //} else  {
    //  SurfaceCommunicator<dim> g_tmp(size);
    //  world.recv(proc_previous(), world_rank, g_tmp);
    //  g_out = g_tmp;
    //}
    //
    //// Distribute g_out
    //for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
    //  thm[i].update_g_out(g_out);

    // Solve the system
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].solve();

  }

  template<int dim>
  void DDM<dim>::prepare_refine(std::vector<std::vector<unsigned int>> connectivity) {

    for (unsigned int i = 0; i < owned_problems.size(); i++)
      thm[i].prepare_mark_interface_for_refinement();

    // Gather refinement:
    for (unsigned int i = 0; i < owned_problems.size(); i++)
      r_in.update(thm[i].return_refine(), owned_problems[i]);

    // distribute r_in to all neighbours:
    if (proc_first()) {
      for (unsigned int problem: owned_problems) {
        for (unsigned int dest: connectivity[problem]) {
          const unsigned int i = domain_map[dest][0];
          RefinementCommunicator<dim> r_tmp(size);

          // send & recieve
          boost::mpi::request reqs[2];
          reqs[0] = world.isend(i, dest, r_in);
          reqs[1] = world.irecv(i, problem, r_tmp);
          boost::mpi::wait_all(reqs, reqs + 2);
          r_in.update(r_tmp, dest);
        }
      }

      for (unsigned int i = 1; i < proc_size(); i++)
        world.send(world_rank + i, world_rank + i, r_in);

    } else {
      RefinementCommunicator<dim> r_tmp(size);
      world.recv(proc_previous(), world_rank, r_tmp);
      r_in = r_tmp;
    }

    for (unsigned int i = 0; i < owned_problems.size(); i++)
      thm[i].update_refine(r_in);

    // mark the cells at the interface for refinment:
    for (unsigned int i = 0; i < owned_problems.size(); i++)
      thm[i].apply_mark_interface_for_refinement();
  }

  template<int dim>
  void DDM<dim>::mark_circular(double radius) {

    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      for (auto &cell : thm[i].return_triangulation().cell_iterators()) 
        if (cell->center().norm() > radius)
          cell->set_refine_flag();
  }

  template<int dim>
  void DDM<dim>::mark_adaptive() {
    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      thm[i].mark_for_refinement_error_estimator();

    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      thm[i].return_triangulation().prepare_coarsening_and_refinement();

    double radius = 0.9;
    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      for (auto &cell : thm[i].return_triangulation().cell_iterators())
        if (cell->center().norm() < radius)
          cell->clear_refine_flag();
  }

  template<int dim>
  void DDM<dim>::mark_circular_coarser(double radius) {
    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      for (auto &cell : thm[i].return_triangulation().cell_iterators()) {
        if (!cell->is_active())
          continue;

        if (cell->center().norm() < radius)
          cell->set_coarsen_flag();
      }
  }

  template<int dim>
  void DDM<dim>::mark_shell_coarser(double inner_radius, double outer_radius) {
    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      for (auto &cell : thm[i].return_triangulation().cell_iterators()) {
        if (!cell->is_active())
          continue;

        if (cell->center().norm() > inner_radius && cell->center().norm() < outer_radius)
          cell->set_coarsen_flag();
      }
  }


    template<int dim>
    void DDM<dim>::fix_stupid_material_id() {
      for ( unsigned int i = 0; i < owned_problems.size(); i++ )
        for (auto &cell : thm[i].return_triangulation().cell_iterators()) {
          if (!cell->is_active())
            continue;

          cell->set_material_id(0);

          bool in_ball = true;
          for ( unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++ )
            if (cell->face(face)->center().norm() > 1) {
              in_ball = false;
              break;
            }

          if ( in_ball )
            cell->set_material_id(1);
        }
    }

  template<int dim>
  void DDM<dim>::refine() {
    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      thm[i].refine();

  }

  template<int dim>
  void DDM<dim>::print_result() const {
    for( unsigned int i = 0; i < owned_problems.size(); i++ ) 
      thm[i].print_results();
  }

  template<int dim>
  bool DDM<dim>::proc_first() const {
    if(world_rank == 0) 
      return true;

    if(proc_id_list[world_rank] == proc_id_list[world_rank - 1])
      return false;

    return true;
  }

  template<int dim>
  unsigned int DDM<dim>::proc_previous() const {
    unsigned int out = 0;
    for(unsigned int i : proc_id_list) {
      if( i == proc_id_list[world_rank] ) 
        return out;
      out++;
    }

    return out;
  }

  template<int dim>
  unsigned int DDM<dim>::proc_size() const {
    unsigned int out = 0;  
    for( unsigned int i : proc_id_list )
      if ( i == proc_id_list[world_rank] )
        out++;

    return out;   
  }

} // namespace: KirasFM



int main(int argc, char *argv[]) {

  try {
    using namespace dealii;
    using namespace KirasFM;
    using namespace KirasFM_Grid_Generator;

    // deal.II MPI interface
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // boost MPI interface
    boost::mpi::environment  env(argc, argv);
    boost::mpi::communicator world;

    ParameterHandler param;
    ParameterReader  prm(param);
    prm.read_parameters("options.prm");

    const unsigned int dim = prm.get_integer (
        "Mesh & geometry parameters", 
        "Dimension" );

    const unsigned int cpus_per_domain = prm.get_integer (
        "MPI parameters",
        "CPUs per domain" );

    const unsigned int slizes = prm.get_integer (
        "MPI parameters",
        "slizes" );

    // Initialize an output, that will be only printed once over all threads,
    // we are initializing this already here, for the late use with domain decomposition
    ConditionalOStream pcout(
      std::cout, 
      (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) 
    );

    TimerOutput timer (
      pcout, 
      TimerOutput::never,
      TimerOutput::wall_times
    );

    // create the connectivity map
    const unsigned int size = (selector == 0) ? slizes : ( (selector == 1) ? slizes * 2 : slizes * 8);
    std::vector<std::vector<unsigned int>> connectivity(size);

    if ( selector == 0 ) {
      for(unsigned int i = 0; i < size; i++) {
        if (i != 0)
          connectivity[i].push_back(i - 1);

        if (i != size - 1)
          connectivity[i].push_back(i + 1);
      }
    }
    else if ( selector == 1 ) {
      for(unsigned int i = 0; i < size; i++) {
        unsigned int layer_id     = i / 2;
        unsigned int subdomain_id = i % 2;

        if (layer_id != 0)
          connectivity[i].push_back(i - 2);

        if ( subdomain_id == 0 )
          connectivity[i].push_back(i + 1);

        if ( subdomain_id == 1 )
          connectivity[i].push_back(i - 1);

        if (layer_id != slizes - 1)
          connectivity[i].push_back(i + 2);
      }
    }
    else if ( selector == 2 ) {
      int neighbor_id[4][2] = {{1,3}, {-1,1}, {-1,1}, {-3,-1}};
      for(unsigned int i = 0; i < size; i++) {

        unsigned int layer_id = i / 8;
        unsigned int subdomain_id = i % 8;

        if (layer_id != 0)
          connectivity[i].push_back(i - 8);

        // inter-layer neighbors
        if (subdomain_id < 4)
          connectivity[i].push_back(i + 4);
        if (subdomain_id >= 4)
          connectivity[i].push_back(i - 4);

        unsigned int halp = (subdomain_id < 4) ? subdomain_id : subdomain_id - 4;
        connectivity[i].push_back(i + neighbor_id[halp][0]);
        connectivity[i].push_back(i + neighbor_id[halp][1]);

        if (layer_id != slizes - 1)
          connectivity[i].push_back(i + 8);

        std::sort(connectivity[i].begin(), connectivity[i].end());
      }
    }
    else {
      Assert(false, ExcInternalError());
    }

//    // --- For debugging ---
//    pcout << "Connectivity map:" << std::endl;
//    for(std::vector<unsigned int> bla : connectivity) {
//      for(unsigned int i : bla)
//        pcout << i << " ";
//      pcout << std::endl;
//    }

//    // --- For debugging ---
//    // print the grid:
//    Triangulation<3> tria;

//    KirasFM_Grid_Generator::KirasFMGridGenerator<3> ddm_gg(0, 3 * 8, 2);
//    std::vector<double> layer_thickness = {2.00, 1.4, 0.6};
//    ddm_gg.create_parted_nano_particle(
//      tria,
//      1.0, /*  radius of the silver ball */
//      layer_thickness
//    );

//    std::ofstream output_file1("Grid1.vtk");
//    GridOut().write_vtk(tria, output_file1);


    pcout << " __    __  __                               ________  __       __ \n" 
          << "/  |  /  |/  |                             /        |/  \\     /  |\n" 
          << "$$ | /$$/ $$/   ______   ______    _______ $$$$$$$$/ $$  \\   /$$ |\n" 
          << "$$ |/$$/  /  | /      \\ /      \\  /       |$$ |__    $$$  \\ /$$$ |\n" 
          << "$$  $$<   $$ |/$$$$$$  |$$$$$$  |/$$$$$$$/ $$    |   $$$$  /$$$$ |\n" 
          << "$$$$$  \\  $$ |$$ |  $$/ /    $$ |$$      \\ $$$$$/    $$ $$ $$/$$ |\n"  
          << "$$ |$$  \\ $$ |$$ |     /$$$$$$$ | $$$$$$  |$$ |      $$ |$$$/ $$ |\n" 
          << "$$ | $$  |$$ |$$ |     $$    $$ |/     $$/ $$ |      $$ | $/  $$ |\n" 
          << "$$/   $$/ $$/ $$/       $$$$$$$/ $$$$$$$/  $$/       $$/      $$/ \n" 
          << std::endl;

    switch ( dim ) {
      case 3: {
          DDM<3> problem(world, prm, pcout, timer, cpus_per_domain, slizes);
          pcout << "==================================================================" << std::endl;
          pcout << "INITIALIZE:" << std::endl;
          problem.initialize();

          for( unsigned int i = 0; i < prm.get_integer("Mesh & geometry parameters", "Number of global iterations"); i++ ) {
//          for( unsigned int i = 0; i < 0; i++ ) {
            pcout << "==================================================================" << std::endl;
            pcout << "STEP " << i + 1 << ":" << std::endl;
            if( false ) {
              problem.mark_adaptive();
              problem.prepare_refine(connectivity);
              problem.prepare_refine(connectivity);
              problem.refine();
            } else {
              problem.step(connectivity);
            }
    
          }
          pcout << "==================================================================" << std::endl;
          problem.print_result();
        } break;

      default:
        Assert(false, ExcNotImplemented());
        break;
    }

  }

  catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
