/*
 * diamond_fin.cc
 *
 *  Created on: Sep 24, 2022
 *      Author: sebastian
 */

#include <kirasfm_grid_generator.h>

namespace KirasFM_Grid_Generator {
  using namespace dealii;

  /*
   * Diamond Fin Waveguide
   */

    // --- Diamond Fin ---
  template<int dim>
  void KirasFMGridGenerator<dim>::set_material_ids_for_diamond_fin(
    Triangulation<dim>& tria,
    const unsigned int half_base_width,
    const unsigned int base_hight,
    const unsigned int fin_hight,
    const unsigned int dz,
	const double       block_size
  ){
     // scaled with 1 = 200nm
     double scale = 1.0;
     const double block_size_z = 1;

     // we assume uniform thick domains
     const unsigned int start      = domain_id * dz * block_size_z;
     const unsigned int end        = (domain_id + 1) * dz * block_size_z;

     { // hat - part 1

       Point<3> bottom_left(
         0,
         (block_size * (base_hight + fin_hight) * scale),
         start * scale
       );
       Point<3> upper_right(
         (2.0 * block_size * half_base_width * scale) + (3 * scale),
   	  (block_size * (base_hight + fin_hight) * scale) + (1 * scale),
   	  end * scale
        );

       mark_block (
         tria,
         bottom_left,
         upper_right,
         2 // material_id
       );
     }

     { // hat - part 2
       Point<3> bottom_left(  block_size * half_base_width * scale, (block_size * (base_hight + fin_hight) * scale) + (1 * scale), start * scale);
       Point<3> upper_right( (block_size * half_base_width * scale) + (3.0 * scale), (block_size * (base_hight + fin_hight) * scale) + (2.75 * scale), end * scale);

       mark_block (
         tria,
         bottom_left,
         upper_right,
         2 // material_id
       );
     }

     { // base
       Point<dim> bottom_left(
         0,
  	   0,
  	   start * scale
  	 );
       Point<dim> upper_right(
         (2.0 * block_size * half_base_width * scale ) + (3.0 * scale),
  	   block_size * base_hight * scale,
  	   end * scale
  	 );

       mark_block (
         tria,
         bottom_left,
         upper_right,
         1 // material_id
       );
     }

    { // Diamond Fin:
      Point<dim> bottom_left(
        (block_size * half_base_width * scale) + (1.0 * scale),
 	    block_size * base_hight * scale,
 	   start * scale
 	 );
      Point<dim> upper_right(
        (block_size * half_base_width * scale) + (2.0 * scale),
 	   (block_size * (base_hight + fin_hight) * scale) + (1.75 * scale),
 	   end * scale
 	 );

      mark_block (
        tria,
        bottom_left,
        upper_right,
        1 // material_id
      );
    }
  }

  template<int dim>
  void KirasFMGridGenerator<dim>::make_diamond_fin(
    Triangulation<dim>& tria,
    const unsigned int base_width_half,
    const unsigned int base_hight,
    const unsigned int fin_hight,
    const unsigned int buffer,
    const unsigned int depth
  ) {
    // step width, i.e. the width of the fin
    // Rescaled with 1/200nm
    const double scale        = 1.0;
    const double block_size   = 1.0;
    const double block_size_z = 1.0;


    // width and hight (number of elements in each direction)
    const unsigned int width = (2 * base_width_half) + 3;
    const unsigned int hight = (base_hight + fin_hight) + 3 + buffer;

    // step width in x direction:
    std::vector<double> dx(width, scale * block_size);
    // now we modify each block, that has not width: scale * block_size
    dx[base_width_half + 0] = 1.00 * scale;
    dx[base_width_half + 1] = 1.00 * scale;
    dx[base_width_half + 2] = 1.00 * scale;

    // step width in y direction:
    std::vector<double> dy(hight, scale * block_size);
    dy[base_hight + fin_hight + 0] = 1.00 * scale;
    dy[base_hight + fin_hight + 1] = 0.75 * scale;
    dy[base_hight + fin_hight + 2] = 1.00 * scale;

    // step width in z direction:
    std::vector<double> dz(depth, scale * block_size_z);


    // we assume uniform thick domains
    const unsigned int start      =  domain_id      * depth * block_size_z * scale;
    const unsigned int end        = (domain_id + 1) * depth * block_size_z * scale;

    // Repetitions
    std::vector<std::vector<double>> repetitions(3);
    repetitions[0] = dx;
    repetitions[1] = dy;
    repetitions[2] = dz;

    // Corners of the grid:
    Point<3> lower_left_front_corner(0, 0, start);
    Point<3> upper_right_back_corner(
      (2.0 * base_width_half * scale * block_size) + (3.00 * scale),
  	( (base_hight + fin_hight + buffer) * scale  * block_size) + (2.75 * scale),
  	end
    );

    GridGenerator::subdivided_hyper_rectangle(
      tria,
      repetitions,
      lower_left_front_corner,
      upper_right_back_corner,
      true
    );

    // Label faces
    set_boundary_ids( tria, 2 /* z-axis */ );

    // Refinement
  //  refine_diamond_fin(tria, base_width_half, base_hight, fin_hight, 4.5, block_size);
    refine_diamond_fin(tria, base_width_half, base_hight, fin_hight, 3.0, block_size);
  //  refine_diamond_fin(tria, base_width_half, base_hight, fin_hight, 1.5, block_size);

    // set material ids
    set_material_ids_for_diamond_fin(
      tria,
      base_width_half,
      base_hight,
      fin_hight,
      depth,
      block_size
    );
  }


  template <int dim>
  void
  KirasFMGridGenerator<dim>::refine_diamond_fin(
    Triangulation<dim>& tria,
    const unsigned int base_width_half,
    const unsigned int base_hight,
    const unsigned int fin_hight,
    double radius,
    double block_size
  ) {

    double center_x = (base_width_half * block_size) + 1.0 + 0.50;
    double center_y = ((base_hight + fin_hight) * block_size) + 1.0 + 0.50;

    for (const auto &cell : tria.cell_iterators()) {
      double distance = std::sqrt( std::pow(cell->center()[0] - center_x, 2) + std::pow(cell->center()[1] - center_y, 2) );
      if( distance < radius )
        cell->set_refine_flag();
    }

    tria.prepare_coarsening_and_refinement();
    tria.execute_coarsening_and_refinement();

  }

} // KirasFM_Grid_Generator

