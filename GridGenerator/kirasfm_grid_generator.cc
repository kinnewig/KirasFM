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

#include "../include/kirasfm_grid_generator.h"

namespace KirasFM_Grid_Generator
{
  using namespace dealii;

  // === Set Material IDs, help functions ===
  // help function: Mark a rectangle in the center of the grid with an different
  // material_id
  template <int dim>
  void
  mark_block(const Triangulation<dim> &triangulation,
             const Point<dim>          bottom_left,
             const Point<dim>          upper_right,
             types::material_id        id)
  {
    Assert(dim == 2 || dim == 3, ExcInternalError());

    Point<dim> center;
    for (auto &cell : triangulation.active_cell_iterators())
      {
        center = cell->center();

        if (dim == 2)
          {
            if (center(0) > bottom_left(0) && center(0) < upper_right(0) &&
                center(1) > bottom_left(1) && center(1) < upper_right(1))
              {
                cell->set_material_id(id);
              }
          }

        else if (dim == 3)
          {
            if (center(0) > bottom_left(0) && center(0) < upper_right(0) &&
                center(1) > bottom_left(1) && center(1) < upper_right(1) &&
                center(2) > bottom_left(2) && center(2) < upper_right(2))
              {
                cell->set_material_id(id);
              }
          }
      }
  }



  // === DDMGridGenerator Class ===
  // --- constructor ---
  template <int dim>
  KirasFMGridGenerator<dim>::KirasFMGridGenerator(
    const unsigned int domain_id,
    const unsigned int N_domains,
    const unsigned int refinements)
    : domain_id(domain_id)
    , N_domains(N_domains)
    , refinements(refinements)
  {}

  // --- Set boundary id's ---
  template <int dim>
  void
  KirasFMGridGenerator<dim>::set_boundary_ids(Triangulation<dim> &in,
                                              const int           axis)
  {
    // axis == 0 -> x
    // axis == 1 -> y
    // axis == 2 -> z

    if (domain_id == 0 && domain_id == N_domains - 1)
      {
        for (auto &cell : in.cell_iterators())
          {
            cell->set_material_id(0);
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 face++)
              {
                if (cell->face(face)->at_boundary())
                  {
                    switch (cell->face(face)->boundary_id() - (2 * axis))
                      {
                        case 0:
                          cell->face(face)->set_boundary_id(1);
                          break;

                          // used to set the boundary data for the analytical
                          // test case where the RHS is not zero
                          // case 3:
                          //  cell->face(face)->set_boundary_id(0);
                          //  break;


                        default:
                          cell->face(face)->set_boundary_id(0); // normaly 0
                          break;
                      } // switch: boundary_id
                  } // fi: at_boundary
              } // rof: face
          } // rof: cell
      }
    else if (domain_id == 0)
      {
        for (auto &cell : in.cell_iterators())
          {
            cell->set_material_id(0);
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 face++)
              {
                if (cell->face(face)->at_boundary())
                  {
                    switch (cell->face(face)->boundary_id() - (2 * axis))
                      {
                        case 0:
                          cell->face(face)->set_boundary_id(1);
                          break;

                        case 1:
                          cell->face(face)->set_boundary_id(domain_id + 3);
                          break;

                        default:
                          cell->face(face)->set_boundary_id(0); // normaly 0
                          break;
                      } // switch: boundary_id
                  } // fi: at_boundary
              } // rof: face
          } // rof: cell
      }
    else if (domain_id == N_domains - 1)
      {
        for (auto &cell : in.cell_iterators())
          {
            cell->set_material_id(0);
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 face++)
              {
                if (cell->face(face)->at_boundary())
                  {
                    switch (cell->face(face)->boundary_id() - (2 * axis))
                      {
                        case 0:
                          cell->face(face)->set_boundary_id(domain_id + 1);
                          break;

                        default:
                          cell->face(face)->set_boundary_id(0); // normaly 0
                          break;
                      } // switch: boundary_id
                  } // fi: at_boundary
              } // rof: face
          } // rof: cell
      }
    else
      {
        for (auto &cell : in.cell_iterators())
          {
            cell->set_material_id(0);
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 face++)
              {
                if (cell->face(face)->at_boundary())
                  {
                    switch (cell->face(face)->boundary_id() - (2 * axis))
                      {
                        case 0:
                          cell->face(face)->set_boundary_id(domain_id + 1);
                          break;

                        case 1:
                          cell->face(face)->set_boundary_id(domain_id + 3);
                          break;

                        default:
                          cell->face(face)->set_boundary_id(0); // normaly 0
                          break;

                      } // switch: boundary_id
                  } // fi: at_boundary
              } // rof: face
          } // rof: cell
      }
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::make_simple_waveguide(Triangulation<dim> &in)
  {
    const double absolut_lower_y = 0.0;
    const double absolut_upper_y = 1.0;

    double step    = (absolut_upper_y - absolut_lower_y) / (1.0 * N_domains);
    double lower_y = step * domain_id;
    double upper_y = step * (domain_id + 1);

    // left lower corner of the rectangle
    const Point<dim> left_edge =
      (dim == 2) ? Point<dim>(0.0, lower_y) : Point<dim>(0.0, lower_y, 0.0);

    // right upper corner of the rectangle
    const Point<dim> right_edge =
      (dim == 2) ? Point<dim>(1.0, upper_y) : Point<dim>(1.0, upper_y, 1.0);

    std::vector<unsigned int> n_subdivisions;
    n_subdivisions.push_back(N_domains);
    n_subdivisions.push_back(1);
    if (dim == 3)
      {
        n_subdivisions.push_back(N_domains);
      }

    // create the rectangle
    GridGenerator::subdivided_hyper_rectangle(
      in, n_subdivisions, left_edge, right_edge, true);

    in.refine_global(refinements);

    set_boundary_ids(in, 1 /* y - axis */);

    // mark the waveguide inside of the grid
    const Point<dim> box_lower_left =
      dim == 2 ? Point<dim>(3.0 / 8.0, 0.0) :
                 Point<dim>(3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0);
    const Point<dim> box_upper_right =
      dim == 2 ? Point<dim>(5.0 / 8.0, 1.0) :
                 Point<dim>(5.0 / 8.0, 5.0 / 8.0, 5.0 / 8.0);
    mark_block<dim>(in, box_lower_left, box_upper_right, 1);
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::refine_circular(
    Triangulation<dim> &in,
    Point<dim>          center,
    unsigned int        axis, /* 0 -> x-axis, 1 -> y-axis, 2 -> z-axis */
    double              radius)
  {
    for (auto &cell : in.cell_iterators())
      {
        double distance = 0;
        for (unsigned int i = 0; i < dim; i++)
          if (i != axis)
            distance = std::pow(cell->center()[i] - center[i], 2);

        // if the cell is inside the radius mark it for refinement
        if (std::sqrt(distance) < radius)
          cell->set_refine_flag();
      }

    //  // prepare the triangulation for refinement,
    //  in.prepare_coarsening_and_refinement();
    //
    //  // actually execute the refinement,
    //  in.execute_coarsening_and_refinement();
  }

  // === Gallium Laser ===
  template <int dim>
  void
  KirasFMGridGenerator<dim>::make_gallium_laser(Triangulation<dim> &tria,
                                                const unsigned int  outer_width,
                                                const unsigned int  inner_width,
                                                const unsigned int  hight,
                                                const unsigned int  depth)
  {
    double block_size = 0.6;
    double thin_layer = 0.1;

    // Create the domain:
    // we assume uniform thick domains
    const double start = domain_id * hight * block_size;
    const double end   = start + (hight * block_size);

    std::vector<double> dx(outer_width + 2, block_size);
    std::vector<double> dy(hight, block_size);
    std::vector<double> dz(depth + 3, block_size);

    dx[1]           = 0.4; // depends on block_size
    dx[outer_width] = 0.4; // depends on block size

    dz[1] = thin_layer;

    // Repetitions
    std::vector<std::vector<double>> repetitions(3);
    repetitions[0] = dx;
    repetitions[1] = dy;
    repetitions[2] = dz;

    // Corners of the grid:
    Point<3> lower_left_front_corner(0, start, 0);
    Point<3> upper_right_back_corner(((outer_width)*block_size) + (2.0 * 0.4),
                                     end,
                                     (depth + 2) * block_size + thin_layer);

    GridGenerator::subdivided_hyper_rectangle(tria,
                                              repetitions,
                                              lower_left_front_corner,
                                              upper_right_back_corner,
                                              true);

    // Set Material Parameters
    double half = (outer_width - inner_width) / 2.0;
    for (const auto &cell : tria.cell_iterators())
      {
        cell->set_material_id(0);
        Point<dim> center = cell->center();

        // mark the cladding with material_id 1
        if (center[2] < block_size)
          {
            cell->set_material_id(1);
          }
        else
          {
            // check if we are on thin or a wide step
            int round_down = center[1] / (2.0 * block_size);
            if (round_down % 2 == 0)
              {
                // wide stepth
                if (center[0] > block_size &&
                    center[0] < (outer_width + 0.4) * block_size)
                  if (center[2] < block_size + thin_layer)
                    cell->set_material_id(2);
                  else if (center[2] < (depth + 1) * block_size + thin_layer)
                    cell->set_material_id(3);
              }
            else
              {
                if (center[0] > (half + 1.0) * block_size &&
                    center[0] < (outer_width + 1 - half) * block_size)
                  if (center[2] > block_size &&
                      center[2] < (depth + 1) * block_size)
                    cell->set_material_id(3);
              }
          }
      }

    // Set Boundary Parameters
    set_boundary_ids(tria, 1 /* y-axis */);
  }


  template class KirasFMGridGenerator<3>;
} // namespace KirasFM_Grid_Generator
