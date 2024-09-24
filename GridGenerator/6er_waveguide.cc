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
   * Laserwritten Waveguide
   *
   * This is a collection of help functions to create the
   * grid, to compare the Maxwell solver to laboratory experiments
   */

  const unsigned int robin     = 0;
  const unsigned int dirichlet = 1;
  const double       TOL       = 0.01;

  // Position depend refrective index.
  // We approximate the refrective index between two points by the following
  // distribution
  class refrective_index
  {
  public:
    refrective_index(const double s,
                     const double r1,
                     const double r2,
                     const double a,
                     const double b)
      : r1(r1)
      , r2(r2)
      , a(a)
      , b(b)
    {
      p_list = {Point<2>({0.3 * s, 0.9 * s}),
                Point<2>({0.7 * s, 0.9 * s}),
                Point<2>({0.1 * s, 0.5 * s}),
                Point<2>({0.9 * s, 0.5 * s}),
                Point<2>({0.3 * s, 0.1 * s}),
                Point<2>({0.7 * s, 0.1 * s})};
    };

    double
    get_refrective_index(const unsigned int i,
                         const double       px,
                         const double       py) const
    {
      if (i != 1)
        {
          double out = r1;
          for (auto p : p_list)
            {
              double x = p.distance(Point<2>({px, py}));
              out += ((r2 - r1) * std::exp(-(std::pow(x - a, 2) / b)));
            }
          return out;
        }

      return 1.0;
    };

  private:
    const double          r1;
    const double          r2;
    const double          a;
    const double          b;
    std::vector<Point<2>> p_list;

  }; // class: refrective index


  // 1. help function: Check if a given point p is inside of the
  // wave guide (return true) or if it is outside of the waveguide
  // (return false)
  template <int dim>
  bool
  in_trapez(Point<dim> p, const double s = 1.0)
  {
    // lets start with the easy part and check y-axis
    if (p[1] > (0.9 * s) || p[1] < (0.1 * s))
      return false;

    // next we check the x-axis, to make things easier we split up the problem:
    if (p[1] < (0.5 * s))
      {
        if (p[0] > ((p[1] / 2.0) + (0.65 * s)))
          return false;

        if (p[0] < ((0.35 * s) - (p[1] / 2.0)))
          return false;
      }
    else
      {
        if (p[0] < ((p[1] / 2.0) - (0.15 * s)))
          return false;

        if (p[0] > ((1.15 * s) - (p[1] / 2.0)))
          return false;
      }


    return true;
  }

  /* A very basic help function to create a cylinder.
   * The build in function GridGenerator::cylinder() can't be used here
   * since the refinement has to match with the refinement of the function
   * GridGenerator::hyper_cube_with_cylindrical_hole.
   */
  template <int dim>
  void
  cylinder(Triangulation<dim> &tria,
           const double        R, // radius
           const double        L  // length
  )
  {
    // define some usefull constants
    const double Rsq = R / std::sqrt(2);
    const double Rh  = R / 2.0;

    // create the centered box:
    const Point<dim>                lower_left(-Rh, -Rh, 0);
    const Point<dim>                upper_right(Rh, Rh, L);
    const std::vector<unsigned int> repetitions = {2, 2, 1};
    GridGenerator::subdivided_hyper_rectangle(tria,
                                              repetitions,
                                              lower_left,
                                              upper_right);

    // define the points for the outer shell:
    const std::vector<double> p0 = {-R, -Rsq, 0, Rsq, R, Rsq, 0, -Rsq};
    const std::vector<double> p1 = {0, Rsq, R, Rsq, 0, -Rsq, -R, -Rsq};
    const std::vector<double> q0 = {-Rh, -Rh, 0, Rh, Rh, Rh, 0, -Rh};
    const std::vector<double> q1 = {0, Rh, Rh, Rh, 0, -Rh, -Rh, -Rh};

    // define the outer shell
    // the left two cells of the outer shell
    for (unsigned int i = 0; i < 2; i++)
      {
        const unsigned int j = (i + 7) % 8;

        const std::vector<Point<dim>> cell = {Point<dim>(p0[j], p1[j], 0),
                                              Point<dim>(q0[j], q1[j], 0),
                                              Point<dim>(p0[i], p1[i], 0),
                                              Point<dim>(q0[i], q1[i], 0),
                                              Point<dim>(p0[j], p1[j], L),
                                              Point<dim>(q0[j], q1[j], L),
                                              Point<dim>(p0[i], p1[i], L),
                                              Point<dim>(q0[i], q1[i], L)};

        Triangulation<dim> tmp_tria;
        GridGenerator::general_cell(tmp_tria, cell);
        GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1.0e-12);

      } // rof: i


    // the upper two cells of the outer shell
    for (unsigned int i = 1; i < 3; i++)
      {
        const unsigned int j = i + 1;

        const std::vector<Point<dim>> cell = {Point<dim>(q0[i], q1[i], 0),
                                              Point<dim>(q0[j], q1[j], 0),
                                              Point<dim>(p0[i], p1[i], 0),
                                              Point<dim>(p0[j], p1[j], 0),
                                              Point<dim>(q0[i], q1[i], L),
                                              Point<dim>(q0[j], q1[j], L),
                                              Point<dim>(p0[i], p1[i], L),
                                              Point<dim>(p0[j], p1[j], L)};

        Triangulation<dim> tmp_tria;
        GridGenerator::general_cell(tmp_tria, cell);
        GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1.0e-12);

      } // rof: i

    // the right two cells of the outer shell
    for (unsigned int i = 3; i < 5; i++)
      {
        const unsigned int j = i + 1;

        const std::vector<Point<dim>> cell = {Point<dim>(q0[j], q1[j], 0),
                                              Point<dim>(p0[j], p1[j], 0),
                                              Point<dim>(q0[i], q1[i], 0),
                                              Point<dim>(p0[i], p1[i], 0),
                                              Point<dim>(q0[j], q1[j], L),
                                              Point<dim>(p0[j], p1[j], L),
                                              Point<dim>(q0[i], q1[i], L),
                                              Point<dim>(p0[i], p1[i], L)};

        Triangulation<dim> tmp_tria;
        GridGenerator::general_cell(tmp_tria, cell);
        GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1.0e-12);

      } // rof: i

    // the lower two cells of the outer shell
    for (unsigned int i = 5; i < 7; i++)
      {
        const unsigned int j = i + 1;

        const std::vector<Point<dim>> cell = {Point<dim>(p0[j], p1[j], 0),
                                              Point<dim>(p0[i], p1[i], 0),
                                              Point<dim>(q0[j], q1[j], 0),
                                              Point<dim>(q0[i], q1[i], 0),
                                              Point<dim>(p0[j], p1[j], L),
                                              Point<dim>(p0[i], p1[i], L),
                                              Point<dim>(q0[j], q1[j], L),
                                              Point<dim>(q0[i], q1[i], L)};

        Triangulation<dim> tmp_tria;
        GridGenerator::general_cell(tmp_tria, cell);
        GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1.0e-12);

      } // rof: i

    for (auto &cell : tria.cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->face(face)->center()[2] == 0 ||
                cell->face(face)->center()[2] == L)
              continue;

            if (cell->face(face)->at_boundary())
              cell->face(face)->set_all_manifold_ids(0);
          }
      }

    Tensor<1, dim>           direction({0.0, 0.0, 1.0});
    CylindricalManifold<dim> manifold(direction, Point<dim>(0.0, 0.0, 0.0));
    tria.set_manifold(0, manifold);

  } // end function: cylinder


  /*
   * Create a hyper cube with a cylinder in the inside,
   * where the core gets marked with material_id = 1.
   * This is usefull as a simple waveguide.
   */
  template <int dim>
  void
  hyper_cube_with_cylindrical_structure(Triangulation<dim> &tria,
                                        Point<dim>          lower_left,
                                        const double        inner_radius,
                                        const double        outer_radius,
                                        const double        L,
                                        types::manifold_id  manifold_id)
  {
    Tensor<1, dim> shift_vector1({lower_left(0) + outer_radius,
                                  lower_left(1) + outer_radius,
                                  lower_left(2)});
    Tensor<1, dim> shift_vector2({lower_left(0) + (outer_radius / 2.0),
                                  lower_left(1) + (outer_radius / 2.0),
                                  lower_left(2)});

    { // create the shell
      Triangulation<dim> tmp_tria;
      GridGenerator::hyper_cube_with_cylindrical_hole(tmp_tria,
                                                      inner_radius,
                                                      outer_radius,
                                                      L);
      GridTools::shift(shift_vector1, tmp_tria);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1.0e-12, true);
    }

    { // create the cylinder in the center of the shell
      Triangulation<dim> tmp_tria;
      cylinder(tmp_tria, inner_radius, L);
      GridTools::shift(shift_vector1, tmp_tria);

      // set the material id of the cylinder
      for (auto &cell : tmp_tria.cell_iterators())
        {
          cell->set_material_id(1);
        }

      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1.0e-12, true);
    }

    for (auto &cell : tria.cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->face(face)->manifold_id() == 0)
              cell->face(face)->set_all_manifold_ids(manifold_id);
          }
      }

    Tensor<1, dim>           direction({0.0, 0.0, 1.0});
    CylindricalManifold<dim> manifold(direction,
                                      Point<dim>(lower_left(0) + (outer_radius),
                                                 lower_left(1) + (outer_radius),
                                                 0.0));
    tria.set_manifold(manifold_id, manifold);
  }


  // We split the creation of the Waveguide into three magic steps, and here
  // follows the magic
  template <int dim>
  void
  magic1(Triangulation<dim> &tria,
         double              L,
         double              lower,
         double              s,
         const unsigned int  refinements,
         const unsigned int  pos,
         const unsigned int  last)
  {
    { // first part:
      std::vector<unsigned int> repetitions = {2, 3, 1};
      Point<dim>                lower_left(0, lower, 0);
      Point<dim>                upper_right(0.2 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
    }

    { // second part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.2 * s, lower + (0.2 * s), 0);
      Point<dim>                upper_right(0.4 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // left hole: (manifold id = 0)
      Triangulation<dim> tmp_tria;
      hyper_cube_with_cylindrical_structure(
        tmp_tria, Point<3>(0.2 * s, lower, 0.0), 0.07 * s, 0.1 * s, L, 0);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // third part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 3, 1};
      Point<dim>                lower_left(0.4 * s, lower, 0);
      Point<dim>                upper_right(0.6 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // fourth part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.6 * s, lower + (0.2 * s), 0);
      Point<dim>                upper_right(0.8 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // right hole: (manifold id = 1)
      Triangulation<dim> tmp_tria;
      hyper_cube_with_cylindrical_structure(
        tmp_tria, Point<3>(0.6 * s, lower, 0.0), 0.07 * s, 0.1 * s, L, 1);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // fifth part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 3, 1};
      Point<dim>                lower_left(0.8 * s, lower, 0);
      Point<dim>                upper_right(1.0 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    // move the grid:
    Tensor<1, dim> shift_vector({0.0, 0.0, L * pos});
    GridTools::shift(shift_vector, tria);

    // set the manifold id:
    Tensor<1, dim>           direction({0.0, 0.0, 1.0});
    CylindricalManifold<dim> manifold_0(
      direction, Point<dim>(0.3 * s, lower + (0.1 * s), 0.0));
    tria.set_manifold(0, manifold_0);

    CylindricalManifold<dim> manifold_1(
      direction, Point<dim>(0.7 * s, lower + (0.1 * s), 0.0));
    tria.set_manifold(1, manifold_1);
    tria.refine_global(refinements);

    // set material_id
    for (auto &cell : tria.cell_iterators())
      {
        if (cell->material_id() == 0)
          {
            if (in_trapez(cell->center(), s))
              {
                cell->set_material_id(2);
              }
          }
      } // rof: cells

    // slize        + in slize position + "+2" shift
    const unsigned int front_face_inc =
      pos == 0 ? dirichlet : ((pos - 1) * 3) + 0 + 2;
    const unsigned int back_face =
      pos == last - 1 ? robin : ((pos + 1) * 3) + 0 + 2;

    // set boundary_ids
    for (auto &cell : tria.cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->face(face)->at_boundary() == false)
              continue;

            // bottom edge
            if (std::abs(cell->face(face)->center()[1] - 0.0) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // left edge
            if (std::abs(cell->face(face)->center()[0] - 0.0) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // right edge
            if (std::abs(cell->face(face)->center()[0] - s) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // front face (incoming -> dirichlet)
            if (std::abs(cell->face(face)->center()[2] - (pos * L)) < TOL)
              cell->face(face)->set_boundary_id(front_face_inc);

            // back face
            if (std::abs(cell->face(face)->center()[2] - ((pos + 1) * L)) < TOL)
              cell->face(face)->set_boundary_id(back_face);

            // upper interface
            if (std::abs(cell->face(face)->center()[1] - (0.3 * s)) < TOL)
              cell->face(face)->set_boundary_id((pos * 3) + 1 + 2);

          } // rof: faces
      } // rof: cells
  }

  template <int dim>
  void
  magic2(Triangulation<dim> &tria,
         double              L,
         double              lower,
         double              s,
         const unsigned int  refinements,
         const unsigned int  pos,
         const unsigned int  last)
  {
    { // lower first part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0, lower, 0);
      Point<dim>                upper_right(0.2 * s, lower + (0.1 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // upper first part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.0 * s, lower + (0.3 * s), 0);
      Point<dim>                upper_right(0.2 * s, lower + (0.4 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // left hole: (manifold id = 0)
      Triangulation<dim> tmp_tria;
      hyper_cube_with_cylindrical_structure(tmp_tria,
                                            Point<3>(0.0,
                                                     lower + (0.1 * s),
                                                     0.0),
                                            0.07 * s,
                                            0.1 * s,
                                            L,
                                            0);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // center part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {6, 4, 1};
      Point<dim>                lower_left(0.2 * s, lower, 0);
      Point<dim>                upper_right(0.8 * s, lower + (0.4 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // lower third part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.8 * s, lower, 0);
      Point<dim>                upper_right(1.0 * s, lower + (0.1 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // upper third part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.8 * s, lower + (0.3 * s), 0);
      Point<dim>                upper_right(1.0 * s, lower + (0.4 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // right hole: (manifold id = 1)
      Triangulation<dim> tmp_tria;
      hyper_cube_with_cylindrical_structure(tmp_tria,
                                            Point<3>(0.8 * s,
                                                     lower + (0.1 * s),
                                                     0.0),
                                            0.07 * s,
                                            0.1 * s,
                                            L,
                                            1);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    // move the grid:
    Tensor<1, dim> shift_vector({0.0, 0.0, L * pos});
    GridTools::shift(shift_vector, tria);

    // set the manifold id:
    Tensor<1, dim>           direction({0.0, 0.0, 1.0});
    CylindricalManifold<dim> manifold_0(
      direction, Point<dim>(0.1 * s, lower + (0.2 * s), 0.0));
    tria.set_manifold(0, manifold_0);

    CylindricalManifold<dim> manifold_1(
      direction, Point<dim>(0.9 * s, lower + (0.2 * s), 0.0));
    tria.set_manifold(1, manifold_1);

    tria.refine_global(refinements);

    // set material_id
    for (auto &cell : tria.cell_iterators())
      {
        if (cell->material_id() == 0)
          {
            if (in_trapez(cell->center(), s))
              {
                cell->set_material_id(2);
              }
          }
      }

    // slize        + neighbour in slize + "+2" shift
    const unsigned int front_face_inc =
      pos == 0 ? dirichlet : ((pos - 1) * 3) + 1 + 2;
    const unsigned int back_face =
      pos == last - 1 ? robin : ((pos + 1) * 3) + 1 + 2;

    // set boundary_ids
    for (auto &cell : tria.cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->face(face)->at_boundary() == false)
              continue;

            // bottom edge
            if (std::abs(cell->face(face)->center()[1] - (0.3 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // left edge
            if (std::abs(cell->face(face)->center()[0] - (0.7 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // right edge
            if (std::abs(cell->face(face)->center()[0] - s) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // front face (incoming -> dirichlet)
            if (std::abs(cell->face(face)->center()[2] - (pos * L)) < TOL)
              cell->face(face)->set_boundary_id(front_face_inc);

            // back face
            if (std::abs(cell->face(face)->center()[2] - ((pos + 1) * L)) < TOL)
              cell->face(face)->set_boundary_id(back_face);

            // lower interface
            if (std::abs(cell->face(face)->center()[1] - (0.3 * s)) < TOL)
              cell->face(face)->set_boundary_id((pos * 3) + 0 + 2);

            // upper interface
            if (std::abs(cell->face(face)->center()[1] - (0.7 * s)) < TOL)
              cell->face(face)->set_boundary_id((pos * 3) + 2 + 2);

          } // rof: faces
      } // rof: cells
  }

  template <int dim>
  void
  magic3(Triangulation<dim> &tria,
         double              L,
         double              lower,
         double              s,
         const unsigned int  refinements,
         const unsigned int  pos,
         const unsigned int  last

  )
  {
    { // first part:
      std::vector<unsigned int> repetitions = {2, 3, 1};
      Point<dim>                lower_left(0, lower, 0);
      Point<dim>                upper_right(0.2 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
    }

    { // second part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.2 * s, lower, 0);
      Point<dim>                upper_right(0.4 * s, lower + (0.1 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // left hole: (manifold id = 0)
      Triangulation<dim> tmp_tria;
      hyper_cube_with_cylindrical_structure(tmp_tria,
                                            Point<3>(0.2 * s,
                                                     lower + (0.1 * s),
                                                     0.0),
                                            0.07 * s,
                                            0.1 * s,
                                            L,
                                            0);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // third part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 3, 1};
      Point<dim>                lower_left(0.4 * s, lower, 0);
      Point<dim>                upper_right(0.6 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // fourth part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 1, 1};
      Point<dim>                lower_left(0.6 * s, lower, 0);
      Point<dim>                upper_right(0.8 * s, lower + (0.1 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // right hole: (manifold id = 1)
      Triangulation<dim> tmp_tria;
      hyper_cube_with_cylindrical_structure(tmp_tria,
                                            Point<3>(0.6 * s,
                                                     lower + (0.1 * s),
                                                     0.0),
                                            0.07 * s,
                                            0.1 * s,
                                            L,
                                            1);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    { // fifth part:
      Triangulation<dim>        tmp_tria;
      std::vector<unsigned int> repetitions = {2, 3, 1};
      Point<dim>                lower_left(0.8 * s, lower, 0);
      Point<dim>                upper_right(1.0 * s, lower + (0.3 * s), L);
      GridGenerator::subdivided_hyper_rectangle(tmp_tria,
                                                repetitions,
                                                lower_left,
                                                upper_right);
      GridGenerator::merge_triangulations(tria, tmp_tria, tria, 1e-10, true);
    }

    // move the grid:
    Tensor<1, dim> shift_vector({0.0, 0.0, L * pos});
    GridTools::shift(shift_vector, tria);

    // set the manifold id:
    Tensor<1, dim>           direction({0.0, 0.0, 1.0});
    CylindricalManifold<dim> manifold_0(
      direction, Point<dim>(0.3 * s, lower + (0.2 * s), 0.0));
    tria.set_manifold(0, manifold_0);

    CylindricalManifold<dim> manifold_1(
      direction, Point<dim>(0.7 * s, lower + (0.2 * s), 0.0));
    tria.set_manifold(1, manifold_1);

    tria.refine_global(refinements);

    // set material_id
    for (auto &cell : tria.cell_iterators())
      {
        if (cell->material_id() == 0)
          {
            if (in_trapez(cell->center(), s))
              {
                cell->set_material_id(2);
              }
          }
      }

    // slize        + neighbour in slize + "+2" shift
    const unsigned int front_face_inc =
      pos == 0 ? dirichlet : ((pos - 1) * 3) + 2 + 2;
    const unsigned int back_face =
      pos == last - 1 ? robin : ((pos + 1) * 3) + 2 + 2;

    // set boundary_ids
    for (auto &cell : tria.cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->face(face)->at_boundary() == false)
              continue;

            // bottom edge
            if (std::abs(cell->face(face)->center()[1] - (0.7 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // left edge
            if (std::abs(cell->face(face)->center()[0] - (1.0 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // right edge
            if (std::abs(cell->face(face)->center()[0] - (1.0 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // front face (incoming -> dirichlet)
            if (std::abs(cell->face(face)->center()[2] - (pos * L)) < TOL)
              cell->face(face)->set_boundary_id(front_face_inc);

            // back face
            if (std::abs(cell->face(face)->center()[2] - ((pos + 1) * L)) < TOL)
              cell->face(face)->set_boundary_id(back_face);

            // lower interface
            if (std::abs(cell->face(face)->center()[1] - (0.7 * s)) < TOL)
              cell->face(face)->set_boundary_id((pos * 3) + 1 + 2);

          } // rof: faces
      } // rof: cells
  } // end function: magic3



  // Everything that is left to do, is to combine magic steps 1 to 3 into the
  // waveguide
  template <int dim>
  void
  KirasFMGridGenerator<dim>::make_6er_waveguide_parted(
    Triangulation<dim> &tria,
    unsigned int        i,
    const unsigned int  refinements,
    const unsigned      last,
    const double        scale)
  {
    switch (i % 3)
      {
        case 0:
          magic1(
            tria, 0.1 * scale, 0.0 * scale, scale, refinements, i / 3, last);
          break;

        case 1:
          magic2(
            tria, 0.1 * scale, 0.3 * scale, scale, refinements, i / 3, last);
          break;

        case 2:
          magic3(
            tria, 0.1 * scale, 0.7 * scale, scale, refinements, i / 3, last);
          break;
      }
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::make_6er_waveguide(Triangulation<dim> &tria,
                                                unsigned int        i,
                                                const unsigned int  refinements,
                                                const unsigned      last,
                                                const double        scale)
  {
    Triangulation<dim> tria_magic1, tria_magic2, tria_magic3;

    magic1(tria_magic1, 0.1 * scale, 0.0 * scale, scale, 0, i, last);
    magic2(tria_magic2, 0.1 * scale, 0.3 * scale, scale, 0, i, last);
    magic3(tria_magic3, 0.1 * scale, 0.7 * scale, scale, 0, i, last);

    GridGenerator::merge_triangulations(
      {&tria_magic1, &tria_magic2, &tria_magic3}, tria, 1e-10, true);

    double       s   = scale;
    double       L   = 0.1 * s;
    unsigned int pos = i;

    const unsigned int front_face_inc = pos == 0 ? dirichlet : (pos - 1) + 2;
    const unsigned int back_face      = pos == last - 1 ? robin : (pos + 1) + 2;

    // set boundary_ids
    for (auto &cell : tria.cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->face(face)->at_boundary() == false)
              continue;

            // bottom edge
            if (std::abs(cell->face(face)->center()[1] - (0.7 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // left edge
            if (std::abs(cell->face(face)->center()[0] - (1.0 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // right edge
            if (std::abs(cell->face(face)->center()[0] - (1.0 * s)) < TOL)
              cell->face(face)->set_boundary_id(robin);

            // front face (incoming -> dirichlet)
            if (std::abs(cell->face(face)->center()[2] - (pos * L)) < TOL)
              cell->face(face)->set_boundary_id(front_face_inc);

            // back face
            if (std::abs(cell->face(face)->center()[2] - ((pos + 1) * L)) < TOL)
              cell->face(face)->set_boundary_id(back_face);

            // lower interface
            if (std::abs(cell->face(face)->center()[1] - (0.7 * s)) < TOL)
              cell->face(face)->set_boundary_id((pos * 3) + 1 + 2);

          } // rof: faces
      } // rof: cells

    tria.refine_global(refinements);
  }
  template class KirasFMGridGenerator<3>;
} // namespace KirasFM_Grid_Generator
