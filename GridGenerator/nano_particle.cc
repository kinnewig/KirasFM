/*
 * nano_particle.cc
 *
 *  Created on: Sep 24, 2022
 *      Author: sebastian
 */

#include <kirasfm_grid_generator.h>

namespace KirasFM_Grid_Generator {
  using namespace dealii;

  /*
   * Nano Particles
   *
   * Cooperation with Antonio Cana Lesina
   *
   * This method provides two function to create nano particles.
   * The first method, creates a rectengular mesh and marks
   * a ball in the center
   * The secound method creates a a sphere and a corresponding
   * rectengular embedding
   */

  // === embedded ball, on a quadratic mesh ===

  // help function: Mark a ball in the center of the grid with an different material_id
  template<int dim>
  void mark_ball (
    const Triangulation<dim> &triangulation,
    const double             radius,
    Point<dim>               center,
    types::material_id       id
  ) {

    Assert(dim == 2 || dim == 3, ExcInternalError());

    for (auto &cell : triangulation.active_cell_iterators()) {

      Point<3> bla(cell->center()[0] - center[0], cell->center()[1] - center[1], cell->center()[2] - center[2]);
      double distance_center = bla.norm();

      if ( distance_center < radius ) {
        cell->set_material_id(id);
      }

    }
  }

  // used for the silver ball in vacuum
  // Creates an rectengular domain and marks a ball in the center of the domain with a
  // different material index.
  template<int dim>
  void KirasFMGridGenerator<dim>::make_embedded_ball( Triangulation<dim>& in ) {

    const double absolut_lower_y = 0.0;
    const double absolut_upper_y = 1.0;

    double step = ( absolut_upper_y - absolut_lower_y ) / ( 1.0 * N_domains );
    double lower_y = step * domain_id;
    double upper_y = step * ( domain_id + 1 );

    // left lower corner of the rectangle
    const Point<dim> left_edge = (dim == 2)
      ? Point<dim>(0.0, lower_y)
      : Point<dim>(0.0, lower_y, 0.0);

    // right upper corner of the rectangle
    const Point<dim> right_edge = (dim == 2)
      ? Point<dim>(1.0, upper_y)
      : Point<dim>(1.0, upper_y, 1.0);

    std::vector<unsigned int> n_subdivisions;
    n_subdivisions.push_back(N_domains);
    n_subdivisions.push_back(1);
    if (dim == 3) {
      n_subdivisions.push_back(N_domains);
    }

    // create the rectangle
    GridGenerator::subdivided_hyper_rectangle(in, n_subdivisions, left_edge, right_edge, true);

    in.refine_global( refinements );

    set_boundary_ids( in, 1 /* y - axis */ );

    mark_ball<dim>(in, 0.25, Point<dim>(0.5,0.5,0.5), 1 );

  }

  // === embedded ball, with spherical manifold ===
  // We create a sphere and than embedded that sphere into a rectengular domain
  // Therefore we first need a few help functions
  // Help Function:

  // 1. Help Function: Quarter_ball_embedding
  // Generate the (quadratic) embedding of the quarter ball,
  // the radius of the ball (a) and the side length of the
  // square (b) can be freely chossen.
  // But the quarterball has laways the center point (0, 0, 0)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::quarter_ball_embedding(
    Triangulation<dim>& tria,
    double outer_radius, /* outer radius */
    double inner_radius  /* inner radius */
  ) {
    const double a = inner_radius * std::sqrt(2.0) / 2e0;
    const double c = a * std::sqrt(3.0) / 2e0;
    const double h = inner_radius / 2e0;

    std::vector<Point<3>> vertices;

    vertices.push_back(Point<3>(0, inner_radius, 0));                      // 0
    vertices.push_back(Point<3>(a, a, 0));                                 // 1
    vertices.push_back(Point<3>(outer_radius, outer_radius, 0));           // 2
    vertices.push_back(Point<3>(0, outer_radius, 0));                      // 3
    vertices.push_back(Point<3>(0, a, a));                                 // 4
    vertices.push_back(Point<3>(c, c, h));                                 // 5
    vertices.push_back(Point<3>(outer_radius, outer_radius, outer_radius));// 6
    vertices.push_back(Point<3>(0, outer_radius, outer_radius));           // 7
    vertices.push_back(Point<3>(inner_radius, 0, 0));                      // 8
    vertices.push_back(Point<3>(outer_radius, 0, 0));                      // 9
    vertices.push_back(Point<3>(a, 0, a));                                 // 10
    vertices.push_back(Point<3>(outer_radius, 0, outer_radius));           // 11
    vertices.push_back(Point<3>(0, 0, inner_radius));                      // 12
    vertices.push_back(Point<3>(0, 0, outer_radius));                      // 13

    const int cell_vertices[3][8] = {
      {0, 1, 3, 2, 4, 5, 7, 6},
      {1, 8, 2, 9, 5, 10, 6, 11},
      {4, 5, 7, 6, 12, 10, 13, 11},
    };
    std::vector<CellData<3>> cells(3);

    for (unsigned int i = 0; i < 3; ++i)
      {
        for (unsigned int j = 0; j < 8; ++j)
          cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
      }

    tria.create_triangulation(vertices,
                              cells,
                              SubCellData()); // no boundary information

  }

  // 2. Help Function: Ball embedding
  // We crate from the quarter ball embedding the (qudratic embedding for a complete ball)
  template <int dim>
  void KirasFMGridGenerator<dim>::ball_embedding (
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius  /* outer radius */
  ) {
    // We create the shell by duplicating the information in each dimension at
    // a time by appropriate rotations, starting from the quarter ball. The
    // rotations make sure we do not generate inverted cells that would appear
    // if we tried the slightly simpler approach to simply mirror the cells.

    Triangulation<dim> tria_piece;
    quarter_ball_embedding(tria_piece, inner_radius, outer_radius);
    const double radius = inner_radius;

    for (unsigned int round = 0; round < dim; ++round)
      {
        Triangulation<dim> tria_copy;
        tria_copy.copy_triangulation(tria_piece);
        tria_piece.clear();
        std::vector<Point<dim>> new_points(tria_copy.n_vertices());
        if (round == 0)
          for (unsigned int v = 0; v < tria_copy.n_vertices(); ++v)
            {
              // rotate by 90 degrees counterclockwise
              new_points[v][0] = -tria_copy.get_vertices()[v][1];
              new_points[v][1] = tria_copy.get_vertices()[v][0];
              if (dim == 3)
                new_points[v][2] = tria_copy.get_vertices()[v][2];
            }
        else if (round == 1)
          {
            for (unsigned int v = 0; v < tria_copy.n_vertices(); ++v)
              {
                // rotate by 180 degrees along the xy plane
                new_points[v][0] = -tria_copy.get_vertices()[v][0];
                new_points[v][1] = -tria_copy.get_vertices()[v][1];
                if (dim == 3)
                  new_points[v][2] = tria_copy.get_vertices()[v][2];
              }
          }
        else if (round == 2)
          for (unsigned int v = 0; v < tria_copy.n_vertices(); ++v)
            {
              // rotate by 180 degrees along the xz plane
              Assert(dim == 3, ExcInternalError());
              new_points[v][0] = -tria_copy.get_vertices()[v][0];
              new_points[v][1] = tria_copy.get_vertices()[v][1];
              new_points[v][2] = -tria_copy.get_vertices()[v][2];
            }
        else
          Assert(false, ExcInternalError());


        // the cell data is exactly the same as before
        std::vector<CellData<dim>> cells;
        cells.reserve(tria_copy.n_cells());
        for (const auto &cell : tria_copy.cell_iterators())
          {
            CellData<dim> data;
            for (unsigned int v : GeometryInfo<dim>::vertex_indices())
              data.vertices[v] = cell->vertex_index(v);
            data.material_id = cell->material_id();
            data.manifold_id = cell->manifold_id();
            cells.push_back(data);
          }

        Triangulation<dim> rotated_tria;
        rotated_tria.create_triangulation(new_points, cells, SubCellData());

        // merge the triangulations - this will make sure that the duplicate
        // vertices in the interior are absorbed
        if (round == dim - 1)
          GridGenerator::merge_triangulations(tria_copy, rotated_tria, tria, 1e-12 * radius);
        else
          GridGenerator::merge_triangulations(tria_copy, rotated_tria, tria_piece, 1e-12 * radius);
      }

    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
        if( cell->face(face)->at_boundary() && cell->face(face)->center().norm_square() < (inner_radius + outer_radius) / 2)
          cell->face(face)->set_all_manifold_ids(0);
      }

    tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0,0,0)));

    tria.refine_global(refinements);
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::hyper_ball_embedded (
    Triangulation<dim> &tria,
    unsigned int index,
    double outer_radius, /* outer radius */
    double inner_radius  /* inner radius */
  ) {

    if ( index == 0 ) {
      double local_inner_radius = inner_radius + 0.3;
      ball_embedding (
        tria,
        outer_radius, /* outer radius */
        local_inner_radius  /* inner radius */
      );

      // --- Set Boundary ID ---
      // Intern interface
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() < (local_inner_radius + outer_radius) / 2)
            cell->face(face)->set_boundary_id(1 + 2);
        }

      // Absorbing boundary condition (Robin boundary condition)
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() > (local_inner_radius + outer_radius) / 2)
            cell->face(face)->set_boundary_id(0);
        }

      // Incoming boundary condition (Dirichlet boundary condition)
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && std::abs(cell->face(face)->center()[0] - outer_radius) < 0.01 )
            cell->face(face)->set_boundary_id(1);
        }

      // --- Set Material ID ---
      // As deal sets very strange material parameter on its own
      // we overite the material parameter first, just to be sure
      for (const auto &cell : tria.cell_iterators())
        cell->set_material_id(0);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, inner_radius, Point<dim>(0.0,0.0,0.0), 1 );

    }
    else if ( index == 1 ) {
  	Triangulation<dim> tria_1;
      GridGenerator::hyper_ball_balanced (
        tria,
        Point<dim>(0,0,0),
  	  inner_radius + 0.3
      );

      tria.refine_global(refinements);

      // --- Set Boundary ID ---
      // Intern interface
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() )
            cell->face(face)->set_boundary_id(0 + 2);
        }

      // --- Set Material ID ---
      // As deal sets very strange material parameter on its own
      // we overite the material parameter first, just to be sure
      for (const auto &cell : tria.cell_iterators())
        cell->set_material_id(0);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, inner_radius, Point<dim>(0.0,0.0,0.0), 1 );

    } else {
  	std::cout << "TRÖÖöÖöOOTTTT" << std::endl;
      // Do nothing
    }

  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::make_embeddded_sphere (
    Triangulation<dim> &tria,
    unsigned int index,
    double outer_radius, /* outer radius */
    double inner_radius  /* inner radius */
  ) {

    // first we compute the thickness of each layer
    const double layer_thickness = outer_radius / N_domains;
    const double start = layer_thickness * domain_id;
    const double end   = layer_thickness * (domain_id + 1);

    // next we need to differ between the three cases: first domain, last domain, every other domain

    // start by the fist domain:
    if ( domain_id == N_domains - 1 ) {

      // --- Create the Grid ---
      ball_embedding (
        tria,
        outer_radius, /* outer radius */
        start         /* inner radius */
      );

      // --- Set Boundary ID ---
      // Intern interface
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() < (start + outer_radius) / 2)
            cell->face(face)->set_boundary_id(domain_id + 2);
        }

      // Absorbing boundary condition (Robin boundary condition)
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() > (start + outer_radius) / 2)
            cell->face(face)->set_boundary_id(0);
        }

      // Incoming boundary condition (Dirichlet boundary condition)
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && std::abs(cell->face(face)->center()[0] - outer_radius) < 0.01 )
            cell->face(face)->set_boundary_id(1);
        }

      // --- Set Material ID ---
      // As deal sets very strange material parameter on its own
      // we overite the material parameter first, just to be sure
      for (const auto &cell : tria.cell_iterators())
        cell->set_material_id(0);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, inner_radius, Point<dim>(0.0,0.0,0.0), 1 );

    }

    // last domain
    else if ( domain_id == 0 ) {

      GridGenerator::hyper_ball_balanced (
        tria,
        Point<dim>(0,0,0),
        end
      );

      tria.refine_global(refinements);

      // --- Set Boundary ID ---
      // Intern interface
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() )
            cell->face(face)->set_boundary_id(domain_id + 1 + 2);
        }

      // --- Set Material ID ---
      // As deal sets very strange material parameter on its own
      // we overite the material parameter first, just to be sure
      for (const auto &cell : tria.cell_iterators())
        cell->set_material_id(0);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, inner_radius, Point<dim>(0.0, 0.0, 0.0), 1 );

    } else {

      GridGenerator::hyper_shell(
	tria,
	Point<dim>(0,0,0),
	start,
	end
      );

      tria.refine_global(refinements);

      // --- Set Boundary ID ---
      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() < (start + end) / 2)
            cell->face(face)->set_boundary_id(domain_id + 2);
        }

      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() > (start + end) / 2)
            cell->face(face)->set_boundary_id(domain_id + 1 + 2);
        }

      // --- Set Material ID ---
      // As deal sets very strange material parameter on its own
      // we overite the material parameter first, just to be sure
      for (const auto &cell : tria.cell_iterators())
        cell->set_material_id(0);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, inner_radius, Point<dim>(0.0, 0.0, 0.0), 1 );
    }

  }

} // KirasFM_Grid_Generator
