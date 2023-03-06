  // === C++ Includes ===
#include <iostream>

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

  //basically this is just a std::vector<bool>, that allows for boos::mpi parallelisation
  class refine_coarse_list {
    public:
      //constructor
      refine_coarse_list();
      refine_coarse_list(unsigned int length);
      refine_coarse_list(std::vector<bool> in);
      refine_coarse_list(const refine_coarse_list& copy);

      std::vector<bool> return_content();

    private:
      friend class boost::serialization::access;

      template<class Archive>
      void serialize(Archive &ar, const unsigned int version) {
        ar& content;
        v = version;
      }

      std::vector<bool> content;
      unsigned int v;
  };

  refine_coarse_list::refine_coarse_list():
    content()
  {}

  refine_coarse_list::refine_coarse_list(const unsigned int length):
    content(length)
  {}

  refine_coarse_list::refine_coarse_list(std::vector<bool> in):
    content(in)
  {}

  refine_coarse_list::refine_coarse_list(const refine_coarse_list& copy):
    content(copy.content)
  {}

  std::vector<bool> refine_coarse_list::return_content() {
    return content;
  }


  
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
      void refine(std::vector<std::vector<unsigned int>> connectivity);

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
    size(slizes * 1),

    domain_map(std::vector<std::vector<unsigned int>>(size)),

    g_out(SurfaceCommunicator<dim>(size)),
    g_in(SurfaceCommunicator<dim>(size))

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
      // Simple Block Benchmark (2D & 3D)
      KirasFM_Grid_Generator::KirasFMGridGenerator<dim> ddm_gg(owned_problems[i], size, refinements);
      //ddm_gg.make_simple_waveguide( thm[i].return_triangulation() );

//      // Diamond Fin (3D only)
//      DDM_Grid_Generator::DDMGridGenerator<dim> ddm_gg(owned_problems[i], size, refinements);
//      ddm_gg.make_diamond_fin(
//        thm[i].return_triangulation(),
//      	  3, // base_width_half
//      	  1, // base_hight
//      	  5, // fin_hight
//      	  1, // buffer height above the fin
//      	  2  // z-depth
//      	);

      /*
       *  Silver ball in vacuum (3D only)
       */
      std::vector<double> layer_thickness = {2.00, 1.625, 1.25, 0.75};
      ddm_gg.create_nano_particle(
        thm[i].return_triangulation(),
        1.0, /*  radius of the silver ball */
        layer_thickness
      );

      ddm_gg.refine_nano_particle(
        thm[i].return_triangulation(),
        1.0
      );

//      // Test Waveguide (3D only)
//        double scale = 1.0;
//        DDM_Grid_Generator::magic_switch( thm[i].return_triangulation() , owned_problems[i] , refinements, slizes, scale);
//        ddm_gg.make_6er_waveguide( thm[i].return_triangulation() , owned_problems[i] , refinements, slizes, scale);

//      // Y-beamsplitter (3D only)
//      Y_Beam_Splitter::y_beamsplitter( thm[i].return_triangulation(), owned_problems[i], refinements, slizes);
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
  void DDM<dim>::refine(std::vector<std::vector<unsigned int>> connectivity) {

    // evaluate the error estimator and mark the corresponding cells for refinement:
    for(unsigned int i = 0; i < owned_problems.size(); i++)
      thm[i].mark_for_refinement();

    //TODO: Add support for mulitple Domains per CPU
    //std::vector<std::vector<bool>>  refine_list = Utilities::MPI::all_gather(MPI_COMM_WORLD, thm[0].return_refine());
    //std::vector<std::vector<bool>>  coarse_list = Utilities::MPI::all_gather(MPI_COMM_WORLD, thm[0].return_coarse());

    //thm[0].update_refine(refine_list);

    // Alternative
    // Gather refinement:
    RefinementCommunicator<dim> rc_in(size);
    for( unsigned int i = 0; i < owned_problems.size(); i++ )
      rc_in.update(thm[i].return_refine(), owned_problems[i]);

    // distribute g_in to all neighbours:
    if( proc_first() ) {
      for(unsigned int problem : owned_problems) {
        for(unsigned int dest : connectivity[problem]) {
          const unsigned int i = domain_map[dest][0];
          RefinementCommunicator<dim> rc_tmp(size);

          // send & recieve
          boost::mpi::request reqs[2];
          reqs[0] = world.isend(i, dest, rc_in);
          reqs[1] = world.irecv(i, problem, rc_tmp);
          boost::mpi::wait_all(reqs, reqs + 2 );
          rc_in.update(rc_tmp, dest);
        }
      }

      for(unsigned int i = 1; i < proc_size(); i++)
        world.send(world_rank + i, world_rank + i, rc_in);

    } else {
      RefinementCommunicator<dim> rc_tmp(size);
      world.recv(proc_previous(), world_rank, rc_tmp);
      rc_in = rc_tmp;
    }

    for(unsigned int i = 0; i < owned_problems.size(); i++)
      thm[i].update_refine(rc_in.refinement());

    // mark the cells at the interface for refinment:
    for(unsigned int i = 0; i < owned_problems.size(); i++)
      thm[i].mark_interface_for_refinement();

    for ( unsigned int i = 0; i < owned_problems.size(); i++ )
      thm[i].refine();

    // initalize the maxwell problems:
    for( unsigned int i = 0; i < owned_problems.size(); i++ )
      thm[i].initialize();

    // we already solve it the first time here:
    for( unsigned int i = 0; i < owned_problems.size(); i++ )
      thm[i].solve();

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
    const unsigned int size = slizes;
    std::vector<std::vector<unsigned int>> connectivity(size);

    //for(unsigned int i = 0; i < size; i++) {
    //  if( i / 3 != 0 )
    //    connectivity[i].push_back(i - 3);

    //  switch( i % 3 ) {
    //    case 0:
    //      connectivity[i].push_back(i + 1);
    //      break;

    //    case 1:
    //      connectivity[i].push_back(i - 1);
    //      connectivity[i].push_back(i + 1);
    //      break;

    //    case 2:
    //      connectivity[i].push_back(i - 1);
    //      break;
    //  }

    //  if( i / 3 != slizes - 1 )
    //    connectivity[i].push_back(i + 3);
    //}

    for(unsigned int i = 0; i < size; i++) {
      if( i  != 0 ) 
        connectivity[i].push_back(i - 1);

      if( i != slizes - 1 ) 
        connectivity[i].push_back(i + 1);
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

//    // For debugging: Show diamond fin
//    DDM_Grid_Generator::DDMGridGenerator<3> ddm_gg(0, 1, 0);
//    ddm_gg.make_diamond_fin (
//      tria,
//      3, // base_width_half
//      1, // base_hight
//      5, // fin_hight
//      1, // buffer height above the fin
//      2  // z-depth
//	  );

//    // Simple Block Benchmark (2D & 3D)
//    DDM_Grid_Generator::DDMGridGenerator<3> ddm_gg(0, 1, 4);
//    ddm_gg.make_simple_waveguide( tria );

//    // For debugging: Silver ball in vacuum
//    KirasFM_Grid_Generator::KirasFMGridGenerator<3> ddm_gg(3, 4, 2);

//    std::vector<double> layer_thickness = {2.00, 1.625, 1.25, 0.75};

//    ddm_gg.hyper_ball_embedded(
//        tria,
//        1.0, /*  radius of the silver ball */
//        layer_thickness
//    );
//
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
            if( i == 4 ) {
              problem.refine(connectivity);
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
