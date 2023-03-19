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

    // 2. Help Function: Quarter_ball_embedding_inverse
    // produces the same grid as if you switch outer_radius and
    // inner_radius in the function quarter_ball_embedding
    // but creates well oriented cells.
    template <int dim>
    void
    KirasFMGridGenerator<dim>::quarter_ball_embedding_inverse(
            Triangulation<dim>& tria,
            double outer_radius, /* outer radius */
            double inner_radius  /* inner radius */
    ) {
      const double a = outer_radius / std::sqrt(2.0);
      const double c = a * std::sqrt(3.0) / 2e0;
      const double h = outer_radius / 2e0;

      std::vector<Point<3>> vertices;

      vertices.push_back(Point<3>(inner_radius, 0, 0));                      // 0 [x]
      vertices.push_back(Point<3>(outer_radius, 0, 0));                      // 1 [x]
      vertices.push_back(Point<3>(a, a, 0));                                 // 2 [x]
      vertices.push_back(Point<3>(inner_radius, inner_radius, 0));           // 3 [x]
      vertices.push_back(Point<3>(inner_radius, 0, inner_radius));           // 4 [x]
      vertices.push_back(Point<3>(a, 0, a));                                 // 5 [x]
      vertices.push_back(Point<3>(c, c, h));                                 // 6 [x]
      vertices.push_back(Point<3>(inner_radius, inner_radius, inner_radius));// 7 [x]
      vertices.push_back(Point<3>(0, inner_radius, 0));                      // 8 [x]
      vertices.push_back(Point<3>(0, outer_radius, 0));                      // 9 [x]
      vertices.push_back(Point<3>(0, 0, outer_radius));                      // 10 [x]
      vertices.push_back(Point<3>(0, a, a));                                 // 11 [x]
      vertices.push_back(Point<3>(0, 0, inner_radius));                      // 12 [x]
      vertices.push_back(Point<3>(0, inner_radius, inner_radius));           // 13 [x]

      const int cell_vertices[3][8] = {
              {0, 1, 3, 2, 4, 5, 7, 6},
              {8, 3, 9, 2, 13, 7, 11, 6},
              {12, 4, 13, 7, 10, 5, 11, 6},
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

    // 3. Help Function: Quarter_shell_embedding
    template <int dim>
    void
    KirasFMGridGenerator<dim>::quarter_shell_embedding(
            Triangulation<dim>& tria,
            double outer_radius, /* outer radius */
            double inner_radius  /* inner radius */
    ) {
      const double a_in = inner_radius * std::sqrt(2.0) / 2e0;
      const double c_in = a_in * std::sqrt(3.0) / 2e0;
      const double h_in = inner_radius / 2e0;

      const double a_out = outer_radius * std::sqrt(2.0) / 2e0;
      const double c_out = a_out * std::sqrt(3.0) / 2e0;
      const double h_out = outer_radius / 2e0;
      std::vector<Point<3>> vertices;

      vertices.push_back(Point<3>(0, inner_radius, 0));                      // 0
      vertices.push_back(Point<3>(a_in, a_in, 0));                                 // 1
      vertices.push_back(Point<3>(a_out, a_out, 0));           // 2
      vertices.push_back(Point<3>(0, outer_radius, 0));                      // 3
      vertices.push_back(Point<3>(0, a_in, a_in));                                 // 4
      vertices.push_back(Point<3>(c_in, c_in, h_in));                                 // 5
      vertices.push_back(Point<3>(c_out, c_out, h_out));// 6
      vertices.push_back(Point<3>(0, a_out, a_out));           // 7
      vertices.push_back(Point<3>(inner_radius, 0, 0));                      // 8
      vertices.push_back(Point<3>(outer_radius, 0, 0));                      // 9
      vertices.push_back(Point<3>(a_in, 0, a_in));                                 // 10
      vertices.push_back(Point<3>(a_out, 0, a_out));           // 11
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

  // 3. Help Function: Ball embedding
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
    quarter_ball_embedding(tria_piece, outer_radius, inner_radius);
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
        if( cell->face(face)->at_boundary() && cell->face(face)->center().norm() < (inner_radius + outer_radius) / 2)
          cell->face(face)->set_all_manifold_ids(0);
      }

    tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0,0,0)));

    tria.refine_global(refinements);
  }


    // 4. Help Function: Hyper Shell
    // We crate from the quarter ball embedding the (qudratic embedding for a complete ball)
    template <int dim>
    void KirasFMGridGenerator<dim>::shell_embedding (
            Triangulation<dim> &tria,
            const double        inner_radius, /* inner radius */
            const double        outer_radius  /* outer radius */
    ) {
      // We create the shell by duplicating the information in each dimension at
      // a time by appropriate rotations, starting from the quarter ball. The
      // rotations make sure we do not generate inverted cells that would appear
      // if we tried the slightly simpler approach to simply mirror the cells.


      Triangulation<dim> tria_piece;
      quarter_shell_embedding(
        tria_piece,
        outer_radius,
        inner_radius
      );

      //Point<dim> center = Point<3>(0.0, 0.0, 0.0);
      //GridGenerator::quarter_hyper_shell(
      //        tria_piece,
      //        center,
      //        inner_radius,
      //        outer_radius
      //);

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
          if( cell->face(face)->at_boundary() )
            cell->face(face)->set_all_manifold_ids(0);
        }

      tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0,0,0)));

      tria.refine_global(refinements);
    }

    // 5. Help Function: Hyper Shell - Filled
    // Create a Hyper Shell with a filled center, i.e. a ball.
    // This is needed to ensure that the cells in the grid numerated accordingly to work
    // with the domain decomposition method.
    template <int dim>
    void KirasFMGridGenerator<dim>::shell_embedding_filled (
            Triangulation<dim> &tria,
            const double        outer_radius  /* outer radius */
    ) {
      // We create the shell by duplicating the information in each dimension at
      // a time by appropriate rotations, starting from the quarter ball. The
      // rotations make sure we do not generate inverted cells that would appear
      // if we tried the slightly simpler approach to simply mirror the cells.

      const double inner_radius = 0.5 * outer_radius;

      // ToDo Add 2D
      Point<dim> center = Point<3>(0.0, 0.0, 0.0);

      Triangulation<dim> tria_piece;
      GridGenerator::quarter_hyper_shell(
              tria_piece,
              center,
              inner_radius,
              outer_radius
      );
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

      // Now fill the center
      Triangulation<dim> center_tria;
      GridGenerator::hyper_ball_balanced(
              center_tria,
              Point<dim>(0,0,0),
              inner_radius
      );
      GridGenerator::merge_triangulations(tria, center_tria, tria, 0.1 * radius);

      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() )
            cell->face(face)->set_all_manifold_ids(0);
        }

      tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0,0,0)));

      tria.refine_global(refinements);
    }

    // 6. Help Function: ball_embedding - Filled
    template <int dim>
    void
    KirasFMGridGenerator<dim>::ball_embedding_inverse (
            Triangulation<dim> &tria,
            const double        outer_radius  /* outer radius */
    ) {
      // We create the shell by duplicating the information in each dimension at
      // a time by appropriate rotations, starting from the quarter ball. The
      // rotations make sure we do not generate inverted cells that would appear
      // if we tried the slightly simpler approach to simply mirror the cells.

      const double inner_radius = 0.3 * outer_radius;

      Triangulation<dim> tria_piece;
      quarter_ball_embedding_inverse (
        tria_piece,
        outer_radius,
        inner_radius
      );
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

      // Now fill the center
      std::vector<std::vector<double>> repetitions = {{inner_radius, inner_radius}, {inner_radius, inner_radius}, {inner_radius, inner_radius}};
      Triangulation<dim> center_tria;
      GridGenerator::subdivided_hyper_rectangle(
              center_tria,
              repetitions,
              Point<dim>(-inner_radius,-inner_radius,-inner_radius),
              Point<dim>(inner_radius,inner_radius,inner_radius)
      );
      GridGenerator::merge_triangulations(tria, center_tria, tria, 0.1 * radius);

      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          if( cell->face(face)->at_boundary() )
            cell->face(face)->set_all_manifold_ids(0);
        }

      tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0,0,0)));

      tria.refine_global(refinements);
    }


    // === Nano Particle ===
    // Summerize the functions defined above to create the Nano Particle
    template<int dim>
    void
    KirasFMGridGenerator<dim>::create_nano_particle (
            Triangulation<dim> &tria,
            double ball_radius,
            std::vector<double> layer_thickness
    ) {
      
      // ToDo: Check the length of layer_thickness (it needs to be of length n_domains)

      if (domain_id == 0) {
        double inner_radius = layer_thickness[domain_id + 1];
        double outer_radius = layer_thickness[domain_id];

        ball_embedding(
                tria,
                inner_radius,
                outer_radius
        );

        // --- Set Boundary ID ---
        // Intern interface
        for (const auto &cell: tria.cell_iterators())
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->center().norm() < (inner_radius + outer_radius) / 2)
              cell->face(face)->set_boundary_id(1 + 2);
          }

        // Absorbing boundary condition (Robin boundary condition)
        for (const auto &cell: tria.cell_iterators())
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->center().norm() > (inner_radius + outer_radius) / 2)
              cell->face(face)->set_boundary_id(0);
          }

        // Incoming boundary condition (Dirichlet boundary condition)
        for (const auto &cell: tria.cell_iterators())
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if (cell->face(face)->at_boundary() && std::abs(cell->face(face)->center()[0] - outer_radius) < 0.01)
              cell->face(face)->set_boundary_id(1);
          }

        // --- Set Material ID ---
        // As deal sets very strange material parameter on its own
        // we overite the material parameter first, just to be sure
        for (const auto &cell: tria.cell_iterators())
          cell->set_material_id(0);

      } else if ( domain_id == N_domains - 1 ) {
         double outer_radius = layer_thickness[domain_id];
         double inner_radius = outer_radius * 0.4;

        ball_embedding_inverse (
                tria,
                outer_radius  /* outer radius */
        );
        //shell_embedding_filled (
        //  tria,
        //  outer_radius
        //);

        // --- Set Boundary ID ---
        for (const auto &cell: tria.cell_iterators())
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->center().norm() > (inner_radius + outer_radius) / 2)
              cell->face(face)->set_boundary_id( (domain_id + 2 ) - 1 );
          }

        // --- Set Material ID ---
        // As deal sets very strange material parameter on its own
        // we overite the material parameter first, just to be sure
        for (const auto &cell: tria.cell_iterators())
          cell->set_material_id(0);

        // mark a ball in the center with a different material id
        mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);

      } else {
        double inner_radius = layer_thickness[domain_id + 1];
        double outer_radius = layer_thickness[domain_id];

        shell_embedding (
          tria,
          inner_radius, /* inner radius */
          outer_radius  /* outer radius */
        );

        // --- Set Boundary ID ---
        for (const auto &cell: tria.cell_iterators())
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->center().norm() < (inner_radius + outer_radius) / 2)
              cell->face(face)->set_boundary_id( (domain_id + 2) + 1 );
          }

        for (const auto &cell: tria.cell_iterators())
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->center().norm() > (inner_radius + outer_radius) / 2)
              cell->face(face)->set_boundary_id( (domain_id + 2 ) - 1 );
          }

        // --- Set Material ID ---
        // As deal sets very strange material parameter on its own
        // we overite the material parameter first, just to be sure
        for (const auto &cell: tria.cell_iterators())
          cell->set_material_id(0);

        // mark a ball in the center with a different material id
        mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);
      }
  }

  template<int dim>
  void KirasFMGridGenerator<dim>::refine_nano_particle (
    Triangulation<dim> &tria,
    double radius_refine,
    double radius_coarsen
  ) {
  for (auto &cell : tria.cell_iterators()) {
    // if the cell is inside the radius mark it for refinement
    if(!cell->is_locally_owned())
      continue;

    if ( cell->center().norm() >= radius_refine )
      cell->set_refine_flag();

    if ( cell->center().norm() <= radius_coarsen )
      cell->set_coarsen_flag();
  }

  // prepare the triangulation for refinement,
  tria.prepare_coarsening_and_refinement();

  // actually execute the refinement,
  tria.execute_coarsening_and_refinement();
}

  template class KirasFMGridGenerator<3>;
} // KirasFM_Grid_Generator
