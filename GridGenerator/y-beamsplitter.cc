/*
 * y-beamsplitter.cc
 *
 *  Created on: Sep 24, 2022
 *      Author: sebastian
 */

#include <kirasfm_grid_generator.h>

namespace material_blocks {
  using namespace dealii;

  template<int dim>
  void mark_block (
    const Triangulation<dim> &triangulation,
    const Point<dim> bottom_left,
    const Point<dim> upper_right,
    types::material_id id
  ) {

    Assert(dim == 2 || dim == 3, ExcInternalError());

    Point<dim> center;
    for (auto &cell : triangulation.active_cell_iterators()) {

      center = cell->center();
  
      if(dim == 2) {
        if (
          center(0) > bottom_left(0)
          && center(0) < upper_right(0)
          && center(1) > bottom_left(1)
          && center(1) < upper_right(1)
        ) {
            cell->set_material_id(id);
        }
      }

      else if(dim == 3) {
        if (
          center(0) > bottom_left(0)
          && center(0) < upper_right(0)
          && center(1) > bottom_left(1)
          && center(1) < upper_right(1)
          && center(2) > bottom_left(2)
          && center(2) < upper_right(2)
        ) {
            cell->set_material_id(id);
        }
      }

    }
  }

  template <int dim>
  void mark_boundary (
    const Triangulation<dim> &triangulation,
    const Point<dim> bottom_left,
    const Point<dim> upper_right,
    types::material_id id
  ) {
    Point<dim> center;
    for (auto &cell : triangulation.active_cell_iterators()) {
      center = cell->center();

      if(dim == 2) {
        if (
          center(0) > bottom_left(0)
          && center(0) < upper_right(0)
          && center(1) > bottom_left(1)
          && center(1) < upper_right(1)
        ) {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if( cell->face(face)->at_boundary() ) {
              cell->face(face)->set_boundary_id(id);
            }
          }
        }
      }

      else if(dim == 3) {
        if (
          center(0) > bottom_left(0)
          && center(0) < upper_right(0)
          && center(1) > bottom_left(1)
          && center(1) < upper_right(1)
          && center(2) > bottom_left(2)
          && center(2) < upper_right(2)
        ) {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
            if( cell->face(face)->at_boundary() ) {
              cell->face(face)->set_boundary_id(id);
            }
          }
        }
      }
    }
  }

  double abs(double in) {
    if (in < 0)
      return -in;

    return in;
  }

  template <int dim>
  void mark_side (
    const Triangulation<dim> &triangulation,
    unsigned int axis, // x = 0, y = 1, z = 2
    double position,
    unsigned int boundary_id,
    double tol
  ) {
    Point<dim> center;
    for (auto &cell : triangulation.cell_iterators()) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
        center = cell->face(face)->center();
        if( abs(center(axis) - position) < tol && cell->face(face)->at_boundary() ) {
          cell->face(face)->set_boundary_id(boundary_id);
        }
      }
    }
  }

  template <int dim>
  void mark_side (
    const Triangulation<dim> &triangulation,
    unsigned int axis, // x = 0, y = 1, z = 2
    Point<dim> p1,
    Point<dim> p2,
    unsigned int boundary_id,
    double tol
  ) {
    Point<dim> center;
    std::vector<unsigned int> other_axis;
    for(unsigned int i = 0; i < dim; i++) {
      if(axis != i)
       other_axis.push_back(i);
    }

    if (triangulation.level > 0)
    for (auto &cell : triangulation.cell_iterators()) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
        center = cell->face(face)->center();
        if( abs(center(axis) - p1(axis)) < tol && cell->face(face)->at_boundary() ) {
          if ( center(other_axis[0]) < p2(other_axis[0]) && center(other_axis[0]) > p1(other_axis[0]) ) {
            if ( center(other_axis[1]) < p2(other_axis[1]) && center(other_axis[1]) > p1(other_axis[1]) ) {
              cell->face(face)->set_boundary_id(boundary_id);
            }
          }
        }
      }
    }

  }

}


namespace KirasFM_Grid_Generator {
  using namespace dealii;

  template<int dim>
  void general_cell_xz(
    Triangulation<dim>& triangulation, 
    std::vector<Point<dim>> cell,
    unsigned int refinements,
    unsigned int material_id = 0
  ) {
    if (refinements == 0) {
      GridGenerator::general_cell(triangulation, cell);
    } else {
      unsigned int real_refinements = 2;

      const double lower_x_steps = (cell[1](0) - cell[0](0)) / (1.0 * real_refinements);
      const double upper_x_steps = (cell[3](0) - cell[2](0)) / (1.0 * real_refinements);
      const double z_steps = (cell[4](2) - cell[0](2)) / (1.0 * real_refinements);

      double lower_x_start = cell[1](0);
      double upper_x_start = cell[3](0);

      const double lower_y = cell[0](1);
      const double upper_y = cell[2](1);

      std::vector< Triangulation<dim> > tria(real_refinements * real_refinements);
      std::vector<Point<dim>> new_cell(8);
      for(unsigned int i = 0; i < real_refinements; i++) {
        double z_start = cell[4](2);
        for(unsigned int j = 0; j < real_refinements; j++) {
          new_cell = {
            Point<dim>(lower_x_start - lower_x_steps, lower_y, z_start - z_steps),
            Point<dim>(lower_x_start, lower_y, z_start - z_steps),
            Point<dim>(upper_x_start - upper_x_steps, upper_y, z_start - z_steps),
            Point<dim>(upper_x_start, upper_y, z_start - z_steps),
            Point<dim>(lower_x_start - lower_x_steps, lower_y, z_start),
            Point<dim>(lower_x_start, lower_y, z_start),
            Point<dim>(upper_x_start - upper_x_steps, upper_y, z_start),
            Point<dim>(upper_x_start, upper_y, z_start)
          };
          general_cell_xz(tria[(i * real_refinements) + j], new_cell, refinements - 1, material_id);
          if( z_start - z_steps >= 0.40 && z_start <= 0.60 ) {
            for (auto &cell : tria[(i * real_refinements) + j].cell_iterators()) 
              cell->set_material_id(material_id);
          }

          z_start -= z_steps;
        }
        lower_x_start -= lower_x_steps;
        upper_x_start -= upper_x_steps;
      }     

      GridGenerator::merge_triangulations(
        //{&tria[0], &tria[1], &tria[2],  &tria[3],  &tria[4],  &tria[5],  &tria[6],  &tria[7], 
        // &tria[8], &tria[9], &tria[10], &tria[11], &tria[12], &tria[13], &tria[14], &tria[15]},
        {&tria[0], &tria[1], &tria[2],  &tria[3]},
        triangulation, 
        0.0001
      );

    }
  }



  template<int dim>
  void y_help_center_lower_part (
    Triangulation<dim>& triangulation, 
    unsigned int refinements,
    double lower_y,
    double hight,
    double center,
    double delta,
    double upper_z,
    double lower_z
  ) { 
    unsigned int real_refinements = 2; 

    if ( refinements == 0 ) {

      std::vector<Point<dim>> cell(8);
      cell = {
        Point<dim>(center - delta, lower_y + hight, lower_z),
        Point<dim>(center, lower_y,                 lower_z),
        Point<dim>(center, lower_y + (1.5 * hight), lower_z), 
        Point<dim>(center + delta, lower_y + hight, lower_z),
        Point<dim>(center - delta, lower_y + hight, upper_z),
        Point<dim>(center, lower_y,                 upper_z),
        Point<dim>(center, lower_y + (1.5 * hight), upper_z), 
        Point<dim>(center + delta, lower_y + hight, upper_z)
      };
      GridGenerator::general_cell(triangulation, cell);

    } else {

      std::vector< Triangulation<dim> > tria(real_refinements);

      double z_step = (upper_z - lower_z) / real_refinements;
      double lower_z_tmp =  lower_z;
      double upper_z_tmp =  lower_z + z_step;

      for(unsigned int i = 0; i < real_refinements; i++) {
        y_help_center_lower_part(
          tria[i],
          refinements - 1,
          lower_y,
          hight,
          center,
          delta,
          upper_z_tmp,
          lower_z_tmp
        );

        lower_z_tmp += z_step;
        upper_z_tmp += z_step;
      }

      GridGenerator::merge_triangulations(
        { &tria[0], &tria[1] },
        triangulation, 
        0.0001
      );
    }
  }

  template<int dim>
  void y_help_center_middle_part (
    Triangulation<dim>& triangulation, 
    unsigned int refinements,
    double lower_y,
    double hight,
    double center,
    double delta,
    double upper_z,
    double lower_z
  ) {
    unsigned int real_refinements = 2;

    if (refinements == 0 ) {
      std::vector<Point<dim>> cell(8);
      std::vector< Triangulation<dim> > tria(2);
      cell = {
        Point<dim>(center - delta, lower_y, lower_z),
        Point<dim>(center, lower_y + (0.5 * hight), lower_z), 
        Point<dim>(center - (2 * delta), lower_y + hight, lower_z),
        Point<dim>(center, lower_y + hight, lower_z),
        Point<dim>(center - delta, lower_y, upper_z),
        Point<dim>(center, lower_y + (0.5 * hight), upper_z), 
        Point<dim>(center - (2 * delta), lower_y + hight, upper_z),
        Point<dim>(center, lower_y + hight, upper_z)
      };
      GridGenerator::general_cell(tria[0], cell);
    
      cell = {
        Point<dim>(center, lower_y + (0.5 * hight), lower_z), 
        Point<dim>(center + delta, lower_y, lower_z),
        Point<dim>(center, lower_y + hight, lower_z),
        Point<dim>(center + (2 * delta), lower_y + hight, lower_z),
        Point<dim>(center, lower_y + (0.5 * hight), upper_z), 
        Point<dim>(center + delta, lower_y, upper_z),
        Point<dim>(center, lower_y + hight, upper_z),
        Point<dim>(center + (2 * delta), lower_y + hight, upper_z),
      };
      GridGenerator::general_cell(tria[1], cell);

      GridGenerator::merge_triangulations(
        { &tria[0], &tria[1] },
        triangulation,
        0.0001
      );

    } else {
      double z_step = (upper_z - lower_z) / real_refinements;
      double lower_z_tmp =  lower_z;
      double upper_z_tmp =  lower_z + z_step;

      std::vector< Triangulation<dim> > tria(real_refinements);

      for(unsigned int i = 0; i < real_refinements; i++) {
        y_help_center_middle_part (
          tria[i], 
          refinements - 1, 
          lower_y, 
          hight, 
          center, 
          delta, 
          upper_z_tmp, 
          lower_z_tmp
        );

        lower_z_tmp += z_step;
        upper_z_tmp += z_step;
      }

      GridGenerator::merge_triangulations(
        { &tria[0], &tria[1] },
        triangulation,
        0.0001
      );

    }
  }

  template<int dim>
  void y_help_center_upper_part (
    Triangulation<dim>& triangulation, 
    unsigned int refinements,
    double lower_y,
    double hight,
    double center,
    double left_c,
    double right_c,
    double delta,
    double upper_z,
    double lower_z
  ) { 
    unsigned int real_refinements = 2;

    if ( refinements == 0 ) {
      std::vector<Point<dim>> cell(8);
      std::vector< Triangulation<dim> > tria(2);
      cell = {
        Point<dim>(left_c, lower_y,                 lower_z),
        Point<dim>(center, lower_y,                 lower_z), 
        Point<dim>(left_c - delta, lower_y + hight, lower_z),
        Point<dim>(center,         lower_y + hight, lower_z),
        Point<dim>(left_c, lower_y,                 upper_z),
        Point<dim>(center, lower_y,                 upper_z), 
        Point<dim>(left_c - delta, lower_y + hight, upper_z),
        Point<dim>(center, lower_y + hight,         upper_z)
      };
      GridGenerator::general_cell(tria[0], cell);

      cell = {
        Point<dim>(center,  lower_y,                 lower_z), 
        Point<dim>(right_c, lower_y,                 lower_z),
        Point<dim>(center, lower_y + hight,          lower_z),
        Point<dim>(right_c + delta, lower_y + hight, lower_z),
        Point<dim>(center, lower_y,                  upper_z), 
        Point<dim>(right_c, lower_y,                 upper_z),
        Point<dim>(center, lower_y + hight,          upper_z),
        Point<dim>(right_c + delta, lower_y + hight, upper_z)
      };
      GridGenerator::general_cell(tria[1], cell);

      GridGenerator::merge_triangulations(
        { &tria[0], &tria[1] },
        triangulation,
        0.0001
      );

    } else {
      std::vector< Triangulation<dim> > tria(2);

      double z_step = (upper_z - lower_z) / real_refinements;
      double lower_z_tmp =  lower_z;
      double upper_z_tmp =  lower_z + z_step;
      for(unsigned int i = 0; i < real_refinements; i++) {
        y_help_center_upper_part (
          tria[i],
          refinements - 1,
          lower_y,
          hight,
          center,
          left_c,
          right_c,
          delta,
          upper_z_tmp,
          lower_z_tmp
        );

        lower_z_tmp += z_step;
        upper_z_tmp += z_step;
      }

      GridGenerator::merge_triangulations(
        { &tria[0], &tria[1] }, 
        triangulation, 
        0.001
      );
    }

  }

  template<int dim>
  void y_help_splitter_lower_part(
    Triangulation<dim>& triangulation,
    const unsigned int refinements,
    const double hight
  ) {
    // initialize
    std::vector< Triangulation<dim> > tria(5);
    std::vector<Point<dim>> cell(8);

    // some more coordinates:
    const double lower_y = 1.2;

    const double lower_z = 0.3;
    const double upper_z = 0.7;

    const double center = 0.5;
    const double left_x = 0.3;
    const double right_x = 0.7;
    
    // compute some usefull variables
    const double delta = 0.1 * hight;

    //waveguide part
    {
      cell = {
        Point<dim>(left_x, lower_y, lower_z),
        Point<dim>(center, lower_y, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(center - delta, lower_y + hight, lower_z),
        Point<dim>(left_x, lower_y, upper_z),
        Point<dim>(center, lower_y, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z),
        Point<dim>(center - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[0], cell, refinements, 1);
      //for (auto &cell : tria[0].active_cell_iterators()) 
      //  cell->set_material_id(1);

      cell = {
        Point<dim>(center , lower_y, lower_z),
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(center  + delta, lower_y + hight, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(center , lower_y, upper_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(center  + delta, lower_y + hight, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[1], cell, refinements, 1);
      //for (auto &cell : tria[1].active_cell_iterators()) 
      //  cell->set_material_id(1);
    }
    //center
    {
    y_help_center_lower_part (
      tria[2],
      refinements,
      lower_y,
      hight,
      center,
      delta,
      upper_z,
      lower_z
    );
    }
    //cladding
    {
      cell = {
        Point<dim>(0.10, lower_y, lower_z),
        Point<dim>(left_x, lower_y, lower_z),
        Point<dim>(0.10, lower_y + hight, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(0.10, lower_y, upper_z),
        Point<dim>(left_x, lower_y, upper_z),
        Point<dim>(0.10, lower_y + hight, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[3], cell, refinements);

      cell = {
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(0.90, lower_y, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(0.90, lower_y + hight, lower_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(0.90, lower_y, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z),
        Point<dim>(0.90, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[4], cell, refinements);
    }

    GridGenerator::merge_triangulations(
      {
        &tria[0], 
        &tria[1], 
        &tria[2],
        &tria[3],
        &tria[4]
      },
      triangulation, 0.001
    );

    //material_blocks::mark_side<dim>(triangulation, 1, 1.2, id + 1, 0.01);
    //material_blocks::mark_side<dim>(triangulation, 1, 1.4, id + 3, 0.01);
  }

  template<int dim>
  void y_help_splitter_middle_part (
    Triangulation<dim>& triangulation,
    const unsigned int refinements,
    const double hight
  ) {
    // compute some usefull variables
    const double delta = 0.1 * hight;

    // some more coordinates:
    const double lower_y = 1.2 + hight;

    const double lower_z = 0.3;
    const double upper_z = 0.7;

    const double center = 0.5;
    const double left_x = 0.3 - delta;
    const double right_x = 0.7 + delta;

    // initialize
    std::vector< Triangulation<dim> > tria(6);
    std::vector<Point<dim>> cell(8);
    //waveguide part
    {
      cell = {
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(center - delta , lower_y, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(center - (2 * delta), lower_y + hight, lower_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(center - delta , lower_y, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z),
        Point<dim>(center - (2 * delta), lower_y + hight, upper_z)
      };
      general_cell_xz(tria[0], cell, refinements, 1);
      //for (auto &cell : tria[0].active_cell_iterators()) 
      //  cell->set_material_id(1);

      cell = {
        Point<dim>(center  + delta, lower_y, lower_z),
        Point<dim>(right_x , lower_y, lower_z),
        Point<dim>(center  + (2 * delta), lower_y + hight, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(center  + delta, lower_y, upper_z),
        Point<dim>(right_x , lower_y, upper_z),
        Point<dim>(center  + (2 * delta), lower_y + hight, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[1], cell, refinements, 1);
      //for (auto &cell : tria[1].active_cell_iterators()) 
      //  cell->set_material_id(1);
    }
    //center
    {
      y_help_center_middle_part(
        tria[2], 
        refinements,
        lower_y,
        hight,
        center,
        delta,
        upper_z,
        lower_z
      );

    }
    //cladding
    {
      cell = {
        Point<dim>(0.10, lower_y, lower_z),
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(0.10, lower_y + hight, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(0.10, lower_y, upper_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(0.10, lower_y + hight, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[3], cell, refinements);

      cell = {
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(0.90, lower_y, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(0.90, lower_y + hight, lower_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(0.90, lower_y, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z),
        Point<dim>(0.90, lower_y + hight, upper_z),
      };
      general_cell_xz(tria[4], cell, refinements);
    }


    GridGenerator::merge_triangulations(
      {
        &tria[0], 
        &tria[1], 
        &tria[2],
        &tria[3],
        &tria[4]
      },
      triangulation, 0.001
    );

  }

  template<int dim>
  void y_help_splitter_upper_part (
    Triangulation<dim>& triangulation,
    const unsigned int refinements,
    const double hight,
    const unsigned int number,
    const double y_start   = 1.2,
    const double x_start_l = 0.3,
    const double x_start_r = 0.7
  ) {
    // compute some usefull variables
    const double delta = 0.1 * hight;

    // some more coordinates:
    const double lower_y = y_start + (number * hight);

    const double lower_z = 0.3;
    const double upper_z = 0.7;

    const double center  = 0.5;
    const double left_c  = x_start_l + 0.2 - (number * delta);
    const double right_c = x_start_r - 0.2 + (number * delta);
    const double left_x  = x_start_l - (number * delta);
    const double right_x = x_start_r + (number * delta);

    // initialize
    std::vector< Triangulation<dim> > tria(5);
    std::vector<Point<dim>> cell(8);
    //waveguide part
    {
      cell = {
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(left_c , lower_y, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(left_c - delta, lower_y + hight, lower_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(left_c , lower_y, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z),
        Point<dim>(left_c - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[0], cell, refinements, 1);
      //for (auto &cell : tria[0].active_cell_iterators()) 
      //  cell->set_material_id(1);

      cell = {
        Point<dim>(right_c, lower_y, lower_z),
        Point<dim>(right_x , lower_y, lower_z),
        Point<dim>(right_c + delta, lower_y + hight, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(right_c, lower_y, upper_z),
        Point<dim>(right_x , lower_y, upper_z),
        Point<dim>(right_c + delta, lower_y + hight, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[1], cell, refinements, 1);
      //for (auto &cell : tria[1].active_cell_iterators()) 
      //  cell->set_material_id(1);
    }

    //center
    {
      y_help_center_upper_part (
        tria[2],
        refinements,
        lower_y,
        hight,
        center,
        left_c,
        right_c,
        delta,
        upper_z,
        lower_z
      );
    }

    //cladding
    {
      cell = {
        Point<dim>(0.10, lower_y, lower_z),
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(0.10, lower_y + hight, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(0.10, lower_y, upper_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(0.10, lower_y + hight, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[3], cell, refinements);

      cell = {
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(0.90, lower_y, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(0.90, lower_y + hight, lower_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(0.90, lower_y, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z),
        Point<dim>(0.90, lower_y + hight, upper_z),
      };
      general_cell_xz(tria[4], cell, refinements);
    }


    GridGenerator::merge_triangulations(
      {
        &tria[0], 
        &tria[1], 
        &tria[2],
        &tria[3],
        &tria[4]
      },
      triangulation, 0.001
    );

  }


  template<int dim>
  void y_help_middle_part (
    Triangulation<dim>& triangulation,
    const unsigned int refinements,
    const double hight,
    const unsigned int number,
    double y_start,
    double x_start_l,
    double x_start_r
  ) {
    // compute some usefull variables
    const double delta = 0.1 * hight;

    // some more coordinates:
    const double lower_y = y_start + (number * hight);

    const double lower_z = 0.30;
    const double upper_z = 0.70;

    const double center  = 0.5;
    const double left_x  = x_start_l - (number * delta);
    const double right_x = x_start_r + (number * delta);

    // initialize
    std::vector< Triangulation<dim> > tria(4);
    std::vector<Point<dim>> cell(8);
    //waveguide part
    {
      cell = {
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(center, lower_y, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(center, lower_y + hight, lower_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(center, lower_y, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z),
        Point<dim>(center, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[0], cell, refinements, 1);

      cell = {
        Point<dim>(center , lower_y, lower_z),
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(center , lower_y + hight, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(center , lower_y, upper_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(center , lower_y + hight, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[1], cell, refinements, 1);
    }

    //cladding
    {
      cell = {
        Point<dim>(0.10, lower_y, lower_z),
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(0.10, lower_y + hight, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(0.10, lower_y, upper_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(0.10, lower_y + hight, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[2], cell, refinements);

      cell = {
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(0.90, lower_y, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(0.90, lower_y + hight, lower_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(0.90, lower_y, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z),
        Point<dim>(0.90, lower_y + hight, upper_z),
      };
      general_cell_xz(tria[3], cell, refinements);
    }


    GridGenerator::merge_triangulations(
      {
        &tria[0], 
        &tria[1], 
        &tria[2],
        &tria[3]
      },
      triangulation, 0.001
    );

  }

  template<int dim>
  void y_help_lower_part (
    Triangulation<dim>& triangulation,
    const unsigned int refinements,
    const double hight,
    const unsigned int number,
    double y_start,
    double x_start_l,
    double x_start_r
  ) {
    // compute some usefull variables
    const double delta = 0.0; // ja ich trolle jetzt zurück!

    // some more coordinates:
    const double lower_y = y_start + (number * hight);

    const double lower_z = 0.30;
    const double upper_z = 0.70;

    const double center  = 0.5;
    const double left_x  = x_start_l - (number * delta);
    const double right_x = x_start_r + (number * delta);

    // initialize
    std::vector< Triangulation<dim> > tria(4);
    std::vector<Point<dim>> cell(8);
    //waveguide part
    {
      cell = {
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(center, lower_y, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(center, lower_y + hight, lower_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(center, lower_y, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z),
        Point<dim>(center, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[0], cell, refinements, 1);

      cell = {
        Point<dim>(center , lower_y, lower_z),
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(center , lower_y + hight, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(center , lower_y, upper_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(center , lower_y + hight, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[1], cell, refinements, 1);
    }

    //cladding
    {
      cell = {
        Point<dim>(0.10, lower_y, lower_z),
        Point<dim>(left_x , lower_y, lower_z),
        Point<dim>(0.10, lower_y + hight, lower_z),
        Point<dim>(left_x - delta, lower_y + hight, lower_z),
        Point<dim>(0.10, lower_y, upper_z),
        Point<dim>(left_x , lower_y, upper_z),
        Point<dim>(0.10, lower_y + hight, upper_z),
        Point<dim>(left_x - delta, lower_y + hight, upper_z)
      };
      general_cell_xz(tria[2], cell, refinements);

      cell = {
        Point<dim>(right_x, lower_y, lower_z),
        Point<dim>(0.90, lower_y, lower_z),
        Point<dim>(right_x + delta, lower_y + hight, lower_z),
        Point<dim>(0.90, lower_y + hight, lower_z),
        Point<dim>(right_x, lower_y, upper_z),
        Point<dim>(0.90, lower_y, upper_z),
        Point<dim>(right_x + delta, lower_y + hight, upper_z),
        Point<dim>(0.90, lower_y + hight, upper_z),
      };
      general_cell_xz(tria[3], cell, refinements);
    }


    GridGenerator::merge_triangulations(
      {
        &tria[0], 
        &tria[1], 
        &tria[2],
        &tria[3]
      },
      triangulation, 0.001
    );

  }

  template<int dim>
  void y_bsp_upper_part(
    Triangulation<dim>& triangulation,
    const unsigned int id,
    const unsigned int refinements,
    double y_start,
    double x_start_l,
    double x_start_r,
    bool last
  ) {
    unsigned int real_refinements = 0;
    switch (refinements) {
      case 2:
        real_refinements = 4;
        break;

      case 3:
        real_refinements = 8;
        break;

      case 4:
        real_refinements = 16;
        break;
    }

    const double hight = 0.2 / real_refinements;
    std::vector< Triangulation<dim> > tria(real_refinements);

    for(unsigned int i = 0; i < real_refinements; i++) {
      y_help_splitter_upper_part(tria[i], refinements, hight, i, y_start, x_start_l, x_start_r);
    }

     switch (refinements) {
      case 2:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3]},
          triangulation, 
          0.001
        );
        break;

      case 3:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7]},
          triangulation, 
          0.001
        );
        break;

      case 4:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7],
           &tria[8], &tria[9], &tria[10], &tria[11], &tria[12], &tria[13], &tria[14], &tria[15]},
          triangulation, 
          0.00001
        );
      break;
    }

    material_blocks::mark_side<dim>(triangulation, 1, y_start, id + 1, 0.001);
    if( last == false )
      material_blocks::mark_side<dim>(triangulation, 1, y_start + 0.2, id + 3, 0.001);

    // for debugging
    // print the grid: 
    //std::ofstream output_file1("Grid.vtk");
    //GridOut().write_vtk(triangulation, output_file1);

  }



  template<int dim>
  void y_bsp_upper_middle (
    Triangulation<dim>& triangulation,
    const unsigned int id,
    const unsigned int refinements
  ) {
    unsigned int real_refinements = 0;
    switch (refinements) {
      case 2:
        real_refinements = 4;
        break;

      case 3:
        real_refinements = 8;
        break;

      case 4:
        real_refinements = 16;
        break;
    }

    const double hight = 0.2 / real_refinements;
    std::vector< Triangulation<dim> > tria(real_refinements);


    y_help_splitter_lower_part ( tria[0], refinements, hight );
    y_help_splitter_middle_part ( tria[1], refinements, hight );
    for(unsigned int i = 2; i < real_refinements; i++) {
      y_help_splitter_upper_part(tria[i], refinements, hight, i);
    }

    switch (refinements) {
      case 2:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3]},
          triangulation, 
          0.001
        );
        break;

      case 3:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7]},
          triangulation, 
          0.001
        );
        break;

      case 4:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7],
           &tria[8], &tria[9], &tria[10], &tria[11], &tria[12], &tria[13], &tria[14], &tria[15]},
          triangulation, 
          0.00001
        );
      break;
    }

    material_blocks::mark_side<dim>(triangulation, 1, 1.2, id + 1, 0.0001);
    material_blocks::mark_side<dim>(triangulation, 1, 1.4, id + 3, 0.0001);

    // print the grid: 
    //std::ofstream output_file1("Grid.vtk");
    //GridOut().write_vtk(triangulation, output_file1);

  }


  template<int dim>
  void y_bsp_lower_middle (
    Triangulation<dim>& triangulation,
    const unsigned int id,
    const unsigned int refinements,
    double y_start,
    double x_start_l,
    double x_start_r
  ) {
    unsigned int real_refinements = 0;
    switch (refinements) {
      case 2:
        real_refinements = 4;
        break;

      case 3:
        real_refinements = 8;
        break;

      case 4:
        real_refinements = 16;
        break;
    }

    const double hight = 0.2 / real_refinements;
    std::vector< Triangulation<dim> > tria(real_refinements);

    for(unsigned int i = 0; i < real_refinements; i++) {
      y_help_middle_part (
        tria[i], 
        refinements, 
        hight, 
        i,
        y_start,
        x_start_l,
        x_start_r
      );
    }

    switch (refinements) {
      case 2:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3]},
          triangulation, 
          0.0001
        );
        break;

      case 3:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7]},
          triangulation, 
          0.0001
        );
        break;

      case 4:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7],
           &tria[8], &tria[9], &tria[10], &tria[11], &tria[12], &tria[13], &tria[14], &tria[15]},
          triangulation, 
          0.0001
        );
        break;

    }

    material_blocks::mark_side<dim>(triangulation, 1, y_start, id + 1, 0.001);
    material_blocks::mark_side<dim>(triangulation, 1, y_start + 0.2, id + 3, 0.001);
  }



  template<int dim>
  void y_bsp_lower_part(
    Triangulation<dim>& triangulation,
    const unsigned int id,
    const unsigned int refinements,
    double y_start,
    double x_start_l,
    double x_start_r
  ) {
    unsigned int real_refinements = 0;
    switch (refinements) {
      case 2:
        real_refinements = 4;
        break;

      case 3:
        real_refinements = 8;
        break;

      case 4:
        real_refinements = 16;
        break;
    }

    const double hight = 0.2 / real_refinements;
    std::vector< Triangulation<dim> > tria(real_refinements);

    for(unsigned int i = 0; i < real_refinements; i++) {
      y_help_lower_part (
        tria[i], 
        refinements, 
        hight, 
        i,
        y_start,
        x_start_l,
        x_start_r
      );
    }

    switch (refinements) {
      case 2:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3]},
          triangulation, 
          0.0001
        );
        break;

      case 3:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7]},
          triangulation, 
          0.0001
        );
        break;

      case 4:
        GridGenerator::merge_triangulations(
          {&tria[0], &tria[1], &tria[2], &tria[3], &tria[4], &tria[5], &tria[6], &tria[7],
           &tria[8], &tria[9], &tria[10], &tria[11], &tria[12], &tria[13], &tria[14], &tria[15]},
          triangulation, 
          0.0001
        );
        break;

    }

    material_blocks::mark_side<dim>(triangulation, 1, y_start, id + 1, 0.001);
    material_blocks::mark_side<dim>(triangulation, 1, y_start + 0.2, id + 3, 0.001);
  }

  template <int dim>
  void
  KirasFMGridGenerator<dim>::y_beamsplitter (
    Triangulation<dim>& tria
  ) {
    double x_left   = 0.4;
    double x_right  = 0.6;
    double y_bottom = 0.0;
    double y_top    = 0.2;
    double update_x = 0.02;

    
    if(domain_id == 0) {
      y_bsp_lower_part(
        tria,
        0, // id
	refinements,
        y_bottom,
        x_left,
        x_right
      );
    } else if (domain_id > 0 && domain_id < 6) {
      y_bottom += (domain_id) * 0.2;
      y_top    += (domain_id) * 0.2;

      x_left  -= (domain_id - 1) * update_x;
      x_right += (domain_id - 1) * update_x;

      y_bsp_lower_middle(
        tria,
        domain_id, 
        refinements,
        y_bottom,
        x_left,
        x_right
      );
    } else if (domain_id == 6) {
      y_bsp_upper_middle(
        tria, 
        6, 
        refinements
      );
    } else if (domain_id > 6 ) {
      bool last = (domain_id == N_domains - 1) ? true : false;
      y_bsp_upper_part(
        tria, 
        domain_id, 
        refinements,
        1.40 + ((domain_id - 7) * 0.20),
        0.28 - ((domain_id - 7) * 0.02),
        0.72 + ((domain_id - 7) * 0.02),
        last
      );
    } // fi: domain
  
  }

}  // namespace KirasFM_Grid_Generator
