/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2024 Sebastian Kinnewig
 *
 * The code is licensed under the GNU Lesser General Public License as
 * published by the Free Software Foundation in version 2.1
 * The full text of the license can be found in the file LICENSE.md
 *
 * ---------------------------------------------------------------------
 * Contact:
 *   Sebastian Kinnewig
 *   Leibniz Universität Hannover (LUH)
 *   Institut für Angewandte Mathematik (IfAM)
 *
 * Questions?
 *   E-Mail: kinnewig@ifam.uni-hannover.de
 *
 * Date: September 2022
 *       Update: September 2024
 */

#include <kirasfm_grid_generator.h>

namespace KirasFM_Grid_Generator
{
  using namespace dealii;

  /*
   * Nano Particles
   *
   * This method provides two function to create nano particles.
   * The first method, creates a rectengular mesh and marks
   * a ball in the center
   * The secound method creates a a sphere and a corresponding
   * rectengular embedding
   */

  // help function: Mark a ball in the center of the grid with an different
  // material_id
  template <int dim>
  void
  mark_ball(const Triangulation<dim> &triangulation,
            const double              radius,
            Point<dim>                center,
            types::material_id        id)
  {
    Assert(dim == 2 || dim == 3, ExcInternalError());

    for (auto &cell : triangulation.active_cell_iterators())
      {
        Point<3> bla(cell->center()[0] - center[0],
                     cell->center()[1] - center[1],
                     cell->center()[2] - center[2]);
        double   distance_center = bla.norm();

        if (distance_center < radius)
          {
            cell->set_material_id(id);
          }
      }
  }


  // ++=============================================================++
  // ||                       Eighth Parts                          ||
  // ++=============================================================++

  // === embedded ball, with spherical manifold ===
  // We create a sphere and than embedded that sphere into a rectengular domain
  // Therefore we first need a few help functions
  // Help Function:

  // Generate the (quadratic) embedding of the quarter ball,
  // the radius of the ball (a) and the side length of the
  // square (b) can be freely chossen.
  // But the quarterball has laways the center point (0, 0, 0)
  template <int dim>
  void
  eighth_ball_embedding(Triangulation<dim> &tria,
                        double              inner_radius, /* inner radius */
                        double              outer_radius  /* outer radius */
  )
  {
    Assert(dim == 3, ExcInternalError());

    const double a = inner_radius * std::sqrt(2.0) / 2e0;
    const double c = a * std::sqrt(3.0) / 2e0;
    const double h = inner_radius / 2e0;

    std::vector<Point<3>> vertices;

    vertices.push_back(Point<3>(0, inner_radius, 0));                       // 0
    vertices.push_back(Point<3>(a, a, 0));                                  // 1
    vertices.push_back(Point<3>(outer_radius, outer_radius, 0));            // 2
    vertices.push_back(Point<3>(0, outer_radius, 0));                       // 3
    vertices.push_back(Point<3>(0, a, a));                                  // 4
    vertices.push_back(Point<3>(c, c, h));                                  // 5
    vertices.push_back(Point<3>(outer_radius, outer_radius, outer_radius)); // 6
    vertices.push_back(Point<3>(0, outer_radius, outer_radius));            // 7
    vertices.push_back(Point<3>(inner_radius, 0, 0));                       // 8
    vertices.push_back(Point<3>(outer_radius, 0, 0));                       // 9
    vertices.push_back(Point<3>(a, 0, a));                       // 10
    vertices.push_back(Point<3>(outer_radius, 0, outer_radius)); // 11
    vertices.push_back(Point<3>(0, 0, inner_radius));            // 12
    vertices.push_back(Point<3>(0, 0, outer_radius));            // 13

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

  // === Quarter_ball_embedding_inverse ===
  // produces the same grid as if you switch outer_radius and
  // inner_radius in the function quarter_ball_embedding
  // but creates well oriented cells.
  template <int dim>
  void
  eighth_ball_embedding_inverse(Triangulation<dim> &tria,
                                double inner_radius, /* inner radius */
                                double outer_radius  /* outer radius */
  )
  {
    const double a = outer_radius / std::sqrt(2.0);
    const double c = a * std::sqrt(3.0) / 2e0;
    const double h = outer_radius / 2e0;

    std::vector<Point<3>> vertices;

    vertices.push_back(Point<3>(inner_radius, 0, 0));            // 0 [x]
    vertices.push_back(Point<3>(outer_radius, 0, 0));            // 1 [x]
    vertices.push_back(Point<3>(a, a, 0));                       // 2 [x]
    vertices.push_back(Point<3>(inner_radius, inner_radius, 0)); // 3 [x]
    vertices.push_back(Point<3>(inner_radius, 0, inner_radius)); // 4 [x]
    vertices.push_back(Point<3>(a, 0, a));                       // 5 [x]
    vertices.push_back(Point<3>(c, c, h));                       // 6 [x]
    vertices.push_back(
      Point<3>(inner_radius, inner_radius, inner_radius));       // 7 [x]
    vertices.push_back(Point<3>(0, inner_radius, 0));            // 8 [x]
    vertices.push_back(Point<3>(0, outer_radius, 0));            // 9 [x]
    vertices.push_back(Point<3>(0, 0, outer_radius));            // 10 [x]
    vertices.push_back(Point<3>(0, a, a));                       // 11 [x]
    vertices.push_back(Point<3>(0, 0, inner_radius));            // 12 [x]
    vertices.push_back(Point<3>(0, inner_radius, inner_radius)); // 13 [x]

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

    // Now fill the center
    Triangulation<dim> center_tria;
    GridGenerator::hyper_rectangle(center_tria,
                                   Point<dim>(0, 0, 0),
                                   Point<dim>(inner_radius,
                                              inner_radius,
                                              inner_radius));
    GridGenerator::merge_triangulations(tria,
                                        center_tria,
                                        tria,
                                        1e-3 * outer_radius);
  }

  // === Help Function: Quarter_shell_embedding ===
  template <int dim>
  void
  eighth_shell_embedding(Triangulation<dim> &tria,
                         double              inner_radius, /* inner radius */
                         double              outer_radius  /* outer radius */
  )
  {
    const double a_in = inner_radius * std::sqrt(2.0) / 2e0;
    const double c_in = a_in * std::sqrt(3.0) / 2e0;
    const double h_in = inner_radius / 2e0;

    const double          a_out = outer_radius * std::sqrt(2.0) / 2e0;
    const double          c_out = a_out * std::sqrt(3.0) / 2e0;
    const double          h_out = outer_radius / 2e0;
    std::vector<Point<3>> vertices;

    vertices.push_back(Point<3>(0, inner_radius, 0));  // 0
    vertices.push_back(Point<3>(a_in, a_in, 0));       // 1
    vertices.push_back(Point<3>(a_out, a_out, 0));     // 2
    vertices.push_back(Point<3>(0, outer_radius, 0));  // 3
    vertices.push_back(Point<3>(0, a_in, a_in));       // 4
    vertices.push_back(Point<3>(c_in, c_in, h_in));    // 5
    vertices.push_back(Point<3>(c_out, c_out, h_out)); // 6
    vertices.push_back(Point<3>(0, a_out, a_out));     // 7
    vertices.push_back(Point<3>(inner_radius, 0, 0));  // 8
    vertices.push_back(Point<3>(outer_radius, 0, 0));  // 9
    vertices.push_back(Point<3>(a_in, 0, a_in));       // 10
    vertices.push_back(Point<3>(a_out, 0, a_out));     // 11
    vertices.push_back(Point<3>(0, 0, inner_radius));  // 12
    vertices.push_back(Point<3>(0, 0, outer_radius));  // 13

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


  // ++=============================================================++
  // ||                        Eighth to *                          ||
  // ++=============================================================++

  // === Eighth to Full ===
  // Colorize:
  // x: 0, 1
  // y: 2, 3
  // z: 4, 5
  // Inner ball : 6
  // outer ball : 7
  template <int dim>
  void
  eighth_to_full(Triangulation<dim> &tria,
                 const double        inner_radius,
                 const double        outer_radius,
                 bool                colorize = true)
  {
    Assert(dim == 3, ExcInternalError());
    const double TOL = 1e-8 * inner_radius;

    for (unsigned int round = 0; round < dim; round++)
      {
        Triangulation<dim> tria_copy;
        tria_copy.copy_triangulation(tria);
        tria.clear();
        std::vector<Point<dim>> new_points(tria_copy.n_vertices());
        if (round == 0)
          for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
            {
              // rotate by 90 degrees counterclockwise
              new_points[v][0] = -tria_copy.get_vertices()[v][1];
              new_points[v][1] = tria_copy.get_vertices()[v][0];
              if (dim == 3)
                new_points[v][2] = tria_copy.get_vertices()[v][2];
            }
        else if (round == 1)
          {
            for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
              {
                // rotate by 180 degrees along the xy plane
                new_points[v][0] = -tria_copy.get_vertices()[v][0];
                new_points[v][1] = -tria_copy.get_vertices()[v][1];
                if (dim == 3)
                  new_points[v][2] = tria_copy.get_vertices()[v][2];
              }
          }
        else if (round == 2)
          for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
            {
              // rotate by 180 degrees along the xz plane
              Assert(dim == 3, ExcInternalError());
              new_points[v][0] = -tria_copy.get_vertices()[v][0];
              new_points[v][1] = tria_copy.get_vertices()[v][1];
              new_points[v][2] = -tria_copy.get_vertices()[v][2];
            }
        else
          {
            Assert(false, ExcInternalError());
          }

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
        GridGenerator::merge_triangulations(tria_copy, rotated_tria, tria, TOL);
      }

    unsigned int x_front = 0, x_back = 0, y_front = 0, y_back = 0, z_front = 0,
                 z_back = 0;
    if (colorize)
      {
        x_front = 0;
        x_back  = 1;
        y_front = 2;
        y_back  = 3;
        z_front = 4;
        z_back  = 5;
      }

    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          // skip all faces, that are not located at the boundary
          if (!cell->face(face)->at_boundary())
            continue;

          // x-direction
          if (std::abs(cell->face(face)->center()[0] + outer_radius) < TOL)
            cell->face(face)->set_boundary_id(x_front);

          else if (std::abs(cell->face(face)->center()[0] - outer_radius) < TOL)
            cell->face(face)->set_boundary_id(x_back);

          // y-direction
          else if (std::abs(cell->face(face)->center()[1] + outer_radius) < TOL)
            cell->face(face)->set_boundary_id(y_front);

          else if (std::abs(cell->face(face)->center()[1] - outer_radius) < TOL)
            cell->face(face)->set_boundary_id(y_back);

          // z-direction
          else if (std::abs(cell->face(face)->center()[2] + outer_radius) < TOL)
            cell->face(face)->set_boundary_id(z_front);

          else if (std::abs(cell->face(face)->center()[2] - outer_radius) < TOL)
            cell->face(face)->set_boundary_id(z_back);

          else if (cell->face(face)->center().norm() <
                   ((3.0 * inner_radius) + outer_radius) / 4.0)
            {
              cell->face(face)->set_all_manifold_ids(0);
              cell->face(face)->set_boundary_id(6);
            }

          else if (cell->face(face)->center().norm() >
                   ((3.0 * inner_radius) + outer_radius) / 4.0)
            {
              cell->face(face)->set_all_manifold_ids(0);
              cell->face(face)->set_boundary_id(7);
            }
        }

    tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0, 0, 0)));
  }

  // === Eighth to Half ===
  // Colorize:
  // x: 0, 1
  // y: 2, 3
  // z: 4, 5
  // Inner ball : 6
  // outer ball : 7
  template <int dim>
  void
  eighth_to_half(Triangulation<dim> &tria,
                 const double        inner_radius,
                 const double        outer_radius,
                 const unsigned int  part,
                 bool                colorize = true)
  {
    AssertIndexRange(part, 2);
    Assert(dim == 3, ExcInternalError());

    const double pi =
      3.1415926535897932384626433832795028841971693993751058209749445923;
    const double TOL = 1e-8 * inner_radius;

    for (unsigned int round = 0; round < dim - 1; round++)
      {
        Triangulation<dim> tria_copy;
        tria_copy.copy_triangulation(tria);
        tria.clear();
        std::vector<Point<dim>> new_points(tria_copy.n_vertices());
        if (round == 0)
          for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
            {
              // rotate by 90 degrees counterclockwise
              new_points[v][0] = -tria_copy.get_vertices()[v][1];
              new_points[v][1] = tria_copy.get_vertices()[v][0];
              if (dim == 3)
                new_points[v][2] = tria_copy.get_vertices()[v][2];
            }
        else if (round == 1)
          {
            for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
              {
                // rotate by 180 degrees along the xy plane
                new_points[v][0] = -tria_copy.get_vertices()[v][0];
                new_points[v][1] = -tria_copy.get_vertices()[v][1];
                if (dim == 3)
                  new_points[v][2] = tria_copy.get_vertices()[v][2];
              }
          }
        else
          {
            Assert(false, ExcInternalError());
          }

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
        GridGenerator::merge_triangulations(tria_copy, rotated_tria, tria, TOL);
      }

    if (part == 1)
      GridTools::rotate(Tensor<1, 3, double>({1.0, 0.0, 0.0}), pi, tria);

    unsigned int x_front = 0, x_back = 0, y_front = 0, y_back = 0, z_front = 0,
                 z_back = 0;
    if (colorize)
      {
        x_front = 0;
        x_back  = 1;
        y_front = 2;
        y_back  = 3;
        z_front = (part == 0) ? 5 : 4;
        z_back  = (part == 0) ? 4 : 5;
      }

    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          // skip all faces, that are not located at the boundary
          if (!cell->face(face)->at_boundary())
            continue;

          // x-direction
          if (std::abs(cell->face(face)->center()[0] + outer_radius) < TOL)
            cell->face(face)->set_boundary_id(x_front);

          else if (std::abs(cell->face(face)->center()[0] - outer_radius) < TOL)
            cell->face(face)->set_boundary_id(x_back);

          // y-direction
          else if (std::abs(cell->face(face)->center()[1] + outer_radius) < TOL)
            cell->face(face)->set_boundary_id(y_front);

          else if (std::abs(cell->face(face)->center()[1] - outer_radius) < TOL)
            cell->face(face)->set_boundary_id(y_back);

          // z-direction
          else if (std::abs(cell->face(face)->center()[2]) < TOL)
            cell->face(face)->set_boundary_id(z_front);

          else if (std::abs(std::abs(cell->face(face)->center()[2]) -
                            outer_radius) < TOL)
            cell->face(face)->set_boundary_id(z_back);

          else if (cell->face(face)->center().norm() <
                   ((3.0 * inner_radius) + outer_radius) / 4.0)
            {
              cell->face(face)->set_all_manifold_ids(0);
              cell->face(face)->set_boundary_id(6);
            }

          else if (cell->face(face)->center().norm() >
                   ((3.0 * inner_radius) + outer_radius) / 4.0)
            {
              cell->face(face)->set_all_manifold_ids(0);
              cell->face(face)->set_boundary_id(7);
            }
        }

    tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0, 0, 0)));
  }

  // === Eighth to eighth ===
  template <int dim>
  void
  eighth_to_eighth(Triangulation<dim> &tria,
                   const double        inner_radius, /* inner radius */
                   const double        outer_radius, /* outer radius */
                   const unsigned int  part,
                   bool                colorize)
  {
    // check that we stay in range
    Assert(dim == 3, ExcInternalError());
    AssertIndexRange(part, 8);

    const double pi =
      3.1415926535897932384626433832795028841971693993751058209749445923;
    const double TOL = 1e-8 * inner_radius;

    if (part > 3)
      GridTools::rotate(Tensor<1, 3, double>({0.0, 0.0, 1.0}), pi / 2.0, tria);

    GridTools::rotate(Tensor<1, 3, double>({1.0, 0.0, 0.0}),
                      ((part % 4) * pi) / 2.0,
                      tria);

    unsigned int x_front = 0, x_back = 0, y_front = 0, y_back = 0, z_front = 0,
                 z_back = 0;
    if (colorize)
      {
        x_front = (part < 4) ? 0 : 1;
        x_back  = (part < 4) ? 1 : 0;
        y_front = (part % 4 == 0 || part % 4 == 3) ? 2 : 3;
        y_back  = (part % 4 == 0 || part % 4 == 3) ? 3 : 2;
        z_front = (part % 4 == 0 || part % 4 == 1) ? 4 : 5;
        z_back  = (part % 4 == 0 || part % 4 == 1) ? 5 : 4;
      }

    // colorize
    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          // skip all faces, that are not located at the boundary
          if (!cell->face(face)->at_boundary())
            continue;

          // x-direction
          if (std::abs(cell->face(face)->center()[0]) < TOL)
            cell->face(face)->set_boundary_id(x_front);

          else if (std::abs(std::abs(cell->face(face)->center()[0]) -
                            outer_radius) < TOL)
            cell->face(face)->set_boundary_id(x_back);

          // y-direction
          else if (std::abs(cell->face(face)->center()[1]) < TOL)
            cell->face(face)->set_boundary_id(y_front);

          else if (std::abs(std::abs(cell->face(face)->center()[1]) -
                            outer_radius) < TOL)
            cell->face(face)->set_boundary_id(y_back);

          // z-direction
          else if (std::abs(cell->face(face)->center()[2]) < TOL)
            cell->face(face)->set_boundary_id(z_front);

          else if (std::abs(std::abs(cell->face(face)->center()[2]) -
                            outer_radius) < TOL)
            cell->face(face)->set_boundary_id(z_back);

          else if (cell->face(face)->center().norm() <
                   ((3.0 * inner_radius) + outer_radius) / 4.0)
            {
              cell->face(face)->set_all_manifold_ids(0);
              cell->face(face)->set_boundary_id(6);
            }

          else if (cell->face(face)->center().norm() >
                   ((3.0 * inner_radius) + outer_radius) / 4.0)
            {
              cell->face(face)->set_all_manifold_ids(0);
              cell->face(face)->set_boundary_id(7);
            }
        }

    tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0, 0, 0)));
  }

  // ++=============================================================++
  // ||                       Mark Layers for DD                    ||
  // ++=============================================================++

  // === mark layers for DD for half ===
  template <int dim>
  void
  mark_layers_for_DD_full(Triangulation<dim> &tria,
                          const unsigned int  domain_id)
  {
    const unsigned int bc_dirichlet = 1;
    const unsigned int bc_robin     = 0;

    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          // skip all faces that are not located at the boundary
          if (!cell->face(face)->at_boundary())
            continue;

          else if (cell->face(face)->boundary_id() == 6)
            cell->face(face)->set_boundary_id(domain_id + 1 + 2);

          else if (cell->face(face)->boundary_id() == 7)
            cell->face(face)->set_boundary_id(domain_id - 1 + 2);

          else if (cell->face(face)->boundary_id() == 4)
            cell->face(face)->set_boundary_id(bc_dirichlet);

          else
            cell->face(face)->set_boundary_id(bc_robin);
        }
  }

  // === mark layers for DD for half ===
  template <int dim>
  void
  mark_layers_for_DD_half(Triangulation<dim> &tria,
                          const unsigned int  domain_id)
  {
    const unsigned int subdomain_id = domain_id % 2;

    unsigned int neighbor    = (subdomain_id == 0) ? 5 : 4;
    int          neighbor_id = (subdomain_id == 0) ? +1 : -1;

    const unsigned int bc_dirichlet = 1;
    const unsigned int bc_robin     = 0;

    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          // skip all faces that are not located at the boundary
          if (!cell->face(face)->at_boundary())
            continue;

          else if (cell->face(face)->boundary_id() == neighbor)
            cell->face(face)->set_boundary_id(domain_id + neighbor_id + 2);

          else if (cell->face(face)->boundary_id() == 6)
            cell->face(face)->set_boundary_id(domain_id + 2 + 2);

          else if (cell->face(face)->boundary_id() == 7)
            cell->face(face)->set_boundary_id(domain_id - 2 + 2);

          else if (subdomain_id == 0 && cell->face(face)->boundary_id() == 4)
            cell->face(face)->set_boundary_id(bc_dirichlet);

          else
            cell->face(face)->set_boundary_id(bc_robin);
        }
  }

  // === mark layers for DDD for eighth ===
  template <int dim>
  void
  mark_layers_for_DD_eighth(Triangulation<dim> &tria,
                            const unsigned int  domain_id)
  {
    const unsigned int subdomain_id = domain_id % 8;

    const unsigned int neighbor_case =
      (subdomain_id < 4) ? subdomain_id : subdomain_id - 4;
    double neighbor[4][2]       = {{2, 4}, {3, 4}, {5, 3}, {5, 2}};
    double neighbor_shift[4][2] = {{1, 3}, {-1, 1}, {-1, 1}, {-3, -1}};

    const unsigned int side           = (subdomain_id < 4) ? 0 : 1;
    const int          side_shift     = (subdomain_id < 4) ? 4 : -4;
    double             outsides[8][3] = {{1, 3, 5},
                                         {1, 2, 5},
                                         {1, 2, 4},
                                         {1, 3, 4},
                                         {0, 3, 5},
                                         {0, 2, 5},
                                         {0, 2, 4},
                                         {0, 3, 4}};

    const unsigned int bc_dirichlet = 1;
    const unsigned int bc_robin     = 0;

    double outsides_associated_ids[8][3] = {
      {bc_robin, bc_robin, bc_robin},
      {bc_robin, bc_robin, bc_robin},
      {bc_robin, bc_robin, bc_dirichlet},
      {bc_robin, bc_robin, bc_dirichlet},
      {bc_robin, bc_robin, bc_robin},
      {bc_robin, bc_robin, bc_robin},
      {bc_robin, bc_robin, bc_dirichlet},
      {bc_robin, bc_robin, bc_dirichlet},
    };

    for (const auto &cell : tria.cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           face++)
        {
          // skip all faces that are not located at the boundary
          if (!cell->face(face)->at_boundary())
            continue;

          if (cell->face(face)->boundary_id() == neighbor[neighbor_case][0])
            cell->face(face)->set_boundary_id(
              domain_id + neighbor_shift[neighbor_case][0] + 2);

          else if (cell->face(face)->boundary_id() ==
                   neighbor[neighbor_case][1])
            cell->face(face)->set_boundary_id(
              domain_id + neighbor_shift[neighbor_case][1] + 2);

          else if (cell->face(face)->boundary_id() == side)
            cell->face(face)->set_boundary_id(domain_id + side_shift + 2);

          else if (cell->face(face)->boundary_id() == 6)
            cell->face(face)->set_boundary_id(domain_id + 8 + 2);

          else if (cell->face(face)->boundary_id() == 7)
            cell->face(face)->set_boundary_id(domain_id - 8 + 2);

          else if (cell->face(face)->boundary_id() == outsides[subdomain_id][0])
            {
              cell->face(face)->set_boundary_id(
                outsides_associated_ids[subdomain_id][0]);
              if (outsides_associated_ids[subdomain_id][2] == bc_robin)
                cell->set_material_id(0);
            }

          else if (cell->face(face)->boundary_id() == outsides[subdomain_id][1])
            {
              cell->face(face)->set_boundary_id(
                outsides_associated_ids[subdomain_id][1]);
              if (outsides_associated_ids[subdomain_id][2] == bc_robin)
                cell->set_material_id(0);
            }

          else if (cell->face(face)->boundary_id() == outsides[subdomain_id][2])
            {
              cell->face(face)->set_boundary_id(
                outsides_associated_ids[subdomain_id][2]);
              if (outsides_associated_ids[subdomain_id][2] == bc_robin)
                cell->set_material_id(0);
            }
        }
  }

  // ++=============================================================++
  // ||                        Full embeddings                      ||
  // ++=============================================================++

  // === Ball embedding ===
  // We crate from the quarter ball embedding the (qudratic embedding for a
  // complete ball)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::ball_embedding(
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius, /* outer radius */
    bool                colorize)
  {
    // Triangulation<dim> tria_piece;
    eighth_ball_embedding(tria, inner_radius, outer_radius);

    eighth_to_full(tria, inner_radius, outer_radius, colorize);

    tria.refine_global(refinements);
  }

  // === Inverse Ball embedding ===
  template <int dim>
  void
  KirasFMGridGenerator<dim>::ball_embedding_inverse(
    Triangulation<dim> &tria,
    const double        outer_radius, /* outer radius */
    bool                colorize)
  {
    const double inner_radius = 0.3 * outer_radius;

    // Triangulation<dim> tria_piece;
    eighth_ball_embedding_inverse(tria, inner_radius, outer_radius);

    eighth_to_full(tria, inner_radius, outer_radius, colorize);

    tria.refine_global(refinements);
  }

  // === Hyper Shell ===
  // We crate from the quarter ball embedding the (qudratic embedding for a
  // complete ball)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::shell_embedding(
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius, /* outer radius */
    bool                colorize)
  {
    // Triangulation<dim> tria_piece;
    eighth_shell_embedding(tria, inner_radius, outer_radius);

    eighth_to_full(tria, inner_radius, outer_radius, colorize);

    tria.refine_global(refinements);
  }

  // ++=============================================================++
  // ||                       Half embeddings                      ||
  // ++=============================================================++

  // === Ball embedding ===
  // We crate from the quarter ball embedding the (qudratic embedding for a
  // complete ball)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::ball_embedding_half(
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius, /* outer radius */
    const unsigned int  part,
    bool                colorize)
  {
    // Triangulation<dim> tria_piece;
    eighth_ball_embedding(tria, inner_radius, outer_radius);

    eighth_to_half(tria, inner_radius, outer_radius, part, colorize);

    tria.refine_global(refinements);
  }

  // === Inverse Ball embedding ===
  template <int dim>
  void
  KirasFMGridGenerator<dim>::ball_embedding_inverse_half(
    Triangulation<dim> &tria,
    const double        outer_radius, /* outer radius */
    const unsigned int  part,
    bool                colorize)
  {
    const double inner_radius = 0.3 * outer_radius;

    // Triangulation<dim> tria_piece;
    eighth_ball_embedding_inverse(tria, inner_radius, outer_radius);

    eighth_to_half(tria, inner_radius, outer_radius, part, colorize);

    tria.refine_global(refinements);
  }

  // === Hyper Shell ===
  // We crate from the quarter ball embedding the (qudratic embedding for a
  // complete ball)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::shell_embedding_half(
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius, /* outer radius */
    const unsigned int  part,
    bool                colorize)
  {
    // Triangulation<dim> tria_piece;
    eighth_shell_embedding(tria, inner_radius, outer_radius);

    eighth_to_half(tria, inner_radius, outer_radius, part, colorize);

    tria.refine_global(refinements);
  }

  // ++=============================================================++
  // ||                      Eighth embeddings                      ||
  // ++=============================================================++

  // === Ball embedding ===
  // We crate from the quarter ball embedding the (qudratic embedding for a
  // complete ball)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::ball_embedding_eighth(
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius, /* outer radius */
    const unsigned int  part,
    bool                colorize)
  {
    // Triangulation<dim> tria_piece;
    eighth_ball_embedding(tria, inner_radius, outer_radius);

    eighth_to_eighth(tria, inner_radius, outer_radius, part, colorize);

    tria.refine_global(refinements);
  }

  // === Inverse Ball embedding ===
  template <int dim>
  void
  KirasFMGridGenerator<dim>::ball_embedding_inverse_eighth(
    Triangulation<dim> &tria,
    const double        outer_radius, /* outer radius */
    const unsigned int  part,
    bool                colorize)
  {
    const double inner_radius = 0.3 * outer_radius;

    // Triangulation<dim> tria_piece;
    eighth_ball_embedding_inverse(tria, inner_radius, outer_radius);

    eighth_to_eighth(tria, inner_radius, outer_radius, part, colorize);

    tria.refine_global(refinements);
  }

  // === Hyper Shell ===
  // We crate from the quarter ball embedding the (qudratic embedding for a
  // complete ball)
  template <int dim>
  void
  KirasFMGridGenerator<dim>::shell_embedding_eighth(
    Triangulation<dim> &tria,
    const double        inner_radius, /* inner radius */
    const double        outer_radius, /* outer radius */
    const unsigned int  part,
    bool                colorize)
  {
    // Triangulation<dim> tria_piece;
    eighth_shell_embedding(tria, inner_radius, outer_radius);

    eighth_to_eighth(tria, inner_radius, outer_radius, part, colorize);

    tria.refine_global(refinements);
  }

  // ++=============================================================++
  // ||                        Create Nano Particle                 ||
  // ++=============================================================++

  // === Nano Particle ===
  // Summerize the functions defined above to create the Nano Particle
  template <int dim>
  void
  KirasFMGridGenerator<dim>::create_nano_particle(
    Triangulation<dim> &tria,
    double              ball_radius,
    std::vector<double> layer_thickness)
  {
    // check that we stay in range
    AssertIndexRange(domain_id, layer_thickness.size());
    Assert(dim == 3, ExcInternalError());

    // Create the most outer shell
    if (domain_id == 0)
      {
        double inner_radius = layer_thickness[domain_id + 1];
        double outer_radius = layer_thickness[domain_id];

        ball_embedding(tria, inner_radius, outer_radius);
        mark_layers_for_DD_full(tria, domain_id);
      }
    else if (domain_id == N_domains - 1)
      {
        double outer_radius = layer_thickness[domain_id];

        ball_embedding_inverse(tria, outer_radius);
        mark_layers_for_DD_full(tria, domain_id);
      }
    else
      {
        double inner_radius = layer_thickness[domain_id + 1];
        double outer_radius = layer_thickness[domain_id];

        shell_embedding(tria, inner_radius, outer_radius);
        mark_layers_for_DD_full(tria, domain_id);
      }

    // mark a ball in the center with a different material id
    mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::create_half_nano_particle(
    Triangulation<dim> &tria,
    double              ball_radius,
    std::vector<double> layer_thickness)
  {
    // This function will be nasty, first of all. each layer will consit out of
    // 8 subdomains.
    const unsigned int layer_id     = domain_id / 2;
    const unsigned int subdomain_id = domain_id % 2;

    // check that we stay in range
    AssertIndexRange(layer_id, layer_thickness.size());
    Assert(dim == 3, ExcInternalError());

    // Create the most outer shell
    if (layer_id == 0)
      {
        double inner_radius = layer_thickness[layer_id + 1];
        double outer_radius = layer_thickness[layer_id];

        ball_embedding_half(tria, inner_radius, outer_radius, subdomain_id);
        mark_layers_for_DD_half(tria, domain_id);
      }

    // create the center
    else if (layer_id == layer_thickness.size() - 1)
      {
        double outer_radius = layer_thickness[layer_id];

        ball_embedding_inverse_half(tria, outer_radius, subdomain_id);
        mark_layers_for_DD_half(tria, domain_id);
      }

    else
      {
        double inner_radius = layer_thickness[layer_id + 1];
        double outer_radius = layer_thickness[layer_id];

        shell_embedding_half(tria, inner_radius, outer_radius, subdomain_id);
        mark_layers_for_DD_half(tria, domain_id);
      }

    // mark a ball in the center with a different material id
    mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::create_eighth_nano_particle(
    Triangulation<dim> &tria,
    double              ball_radius,
    std::vector<double> layer_thickness)
  {
    // This function will be nasty, first of all. each layer will consit out of
    // 8 subdomains.
    const unsigned int layer_id     = domain_id / 8;
    const unsigned int subdomain_id = domain_id % 8;

    // check that we stay in range
    AssertIndexRange(layer_id, layer_thickness.size());
    Assert(dim == 3, ExcInternalError());

    // Create the most outer shell
    if (layer_id == 0)
      {
        double inner_radius = layer_thickness[layer_id + 1];
        double outer_radius = layer_thickness[layer_id];

        ball_embedding_eighth(tria, inner_radius, outer_radius, subdomain_id);
        mark_layers_for_DD_eighth(tria, domain_id);
      }

    // create the center
    else if (layer_id == layer_thickness.size() - 1)
      {
        double outer_radius = layer_thickness[layer_id];

        ball_embedding_inverse_eighth(tria, outer_radius, subdomain_id);
        mark_layers_for_DD_eighth(tria, domain_id);
      }

    else
      {
        double inner_radius = layer_thickness[layer_id + 1];
        double outer_radius = layer_thickness[layer_id];

        shell_embedding_eighth(tria, inner_radius, outer_radius, subdomain_id);
        mark_layers_for_DD_eighth(tria, domain_id);
      }

    // mark a ball in the center with a different material id
    mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);
  }

  template class KirasFMGridGenerator<3>;
} // namespace KirasFM_Grid_Generator
