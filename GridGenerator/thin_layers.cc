/*
 * thin_layers.cc
 *
 *  Created on: Sep 24, 2022
 *      Author: sebastian
 */

#include <kirasfm_grid_generator.h>

namespace KirasFM_Grid_Generator {
  using namespace dealii;

  /*
   *  Stack of thin layers
   *
   *  Cooperation with Anna Karoline Rüssler @ LZH
   *  a.ruesseler@lzh.de
   *
   *  Description of a stack of many thin layers, we provide a list which contains
   *  the material (presented by an index) and the thickniss of each layer
   */

  // This is not the nice english way, but at this point, I just declare
  // the vector with the different materials and layer thickness and the length of
  // the vector
  unsigned int intern_n_layers = 109;
  std::vector<std::vector<double>> intern_material_data_vector = {
    {0, 177.6020},
    {1, 167.2393},
    {0, 214.4038},
    {1, 167.7453},
    {0, 216.0531},
    {1, 167.2408},
    {0, 228.2493},
    {1, 225.4031},
    {0, 221.7379},
    {1, 177.9165},
    {0, 215.4055},
    {1, 176.7674},
    {0, 187.2404},
    {1, 625.4202},
    {0, 176.6029},
    {1, 184.7640},
    {0, 179.8806},
    {1, 173.0282},
    {0, 193.8546},
    {1, 159.6871},
    {0, 209.2479},
    {1, 159.5480},
    {0, 264.7007},
    {1, 165.5092},
    {0, 231.1213},
    {1, 169.0884},
    {0, 209.4750},
    {1, 184.7163},
    {0, 182.1187},
    {1, 180.3101},
    {0, 172.0546},
    {1, 206.8213},
    {0, 177.0614},
    {1, 197.9979},
    {0, 187.6925},
    {1, 152.4299},
    {0, 197.4258},
    {1, 136.3725},
    {0, 218.9817},
    {1, 156.2646},
    {0, 276.0325},
    {1, 163.5847},
    {0, 230.9133},
    {1, 146.8633},
    {0, 197.1767},
    {1, 180.5293},
    {0, 171.5620},
    {1, 197.8482},
    {0, 188.9392},
    {1, 216.1456},
    {0, 197.0956},
    {1, 208.3540},
    {0, 172.1632},
    {1, 193.7602},
    {0, 165.6477},
    {1, 187.0357},
    {0, 203.4977},
    {1, 594.8087},
    {0, 291.8020},
    {1, 161.8725},
    {0, 229.5809},
    {1, 144.9028},
    {0, 175.8569},
    {1, 180.2684},
    {0, 457.0883},
    {1, 653.7643},
    {0, 187.1554},
    {1, 165.6238},
    {0, 197.5501},
    {1, 198.3909},
    {0, 197.6444},
    {1, 158.8728},
    {0, 231.5581},
    {1, 134.9288},
    {0, 247.5977},
    {1, 178.7551},
    {0, 207.3621},
    {1, 280.4328},
    {0, 176.3225},
    {1, 693.8048},
    {0, 140.2536},
    {1, 134.6405},
    {0, 108.3655},
    {1, 165.3079},
    {0, 79.4600},
    {1, 188.1386},
    {0, 144.0575},
    {1, 141.1309},
    {0, 197.2583},
    {1, 132.1464},
    {0, 94.2134},
    {1, 130.5587},
    {0, 139.9587},
    {1, 184.0310},
    {0, 165.9755},
    {1, 208.0514},
    {0, 132.7817},
    {1, 196.1951},
    {0, 114.5408},
    {1, 183.2910},
    {0, 109.6062},
    {1, 496.7625},
    {0, 100.1729},
    {1, 184.0713},
    {0, 551.8215},
    {1, 180.4197},
    {0, 229.2978},
    {1, 196.0905},
    {0, 400.8203}
  };


  // Help function:
  template<int dim>
  void KirasFMGridGenerator<dim>::help_make_thin_layer (
    Triangulation<dim> &tria,
    const unsigned int              steps_x,         /* number of cells in x-direction */
    const unsigned int              steps_y,         /* number of cells in y-direction */
    const double                    block_size_xy,   /* cell size in x and y directions */
    const std::vector<std::vector<double>> dz_list,  /* cell size in x and y directions */
    const std::vector<double>       material_list    /* List of thickniss of each layer */
  ) {

    // We begin by computing which part does this subdomain contains
    double start = dz_list[domain_id].front();
    double end   = dz_list[domain_id].back();

    // Corners of the grid:
    Point<3> lower_left_front_corner(0, 0, start);
    Point<3> upper_right_back_corner(steps_x * block_size_xy, steps_y * block_size_xy, end);

    // Repetitions
    std::vector<double> dx(steps_x, block_size_xy);
    std::vector<double> dy(steps_y, block_size_xy);
    std::vector<double> dz( dz_list[domain_id] );

    std::vector<std::vector<double>> repetitions = {dx, dy, dz_list[domain_id] };

    // Create the grid
    GridGenerator::subdivided_hyper_rectangle(
      tria,                    // Triangulation
      repetitions,
      lower_left_front_corner,
      upper_right_back_corner,
      false
    );

    // Set Boundary IDs
    set_boundary_ids( tria, 2 /* z-direction */ );

    // Set Material IDs
    for (const auto &cell : tria.cell_iterators()) {
      double current_pos = cell->center()[2];
      unsigned int i = 0;
      for (double p : dz_list[domain_id] ) {
        if ( p > current_pos )
          break;
        i++;
      }
      cell->set_material_id(material_list[i]);
    }

  }


  // Function that
  template<int dim>
  void KirasFMGridGenerator<dim>::make_thin_layers(
    Triangulation<dim> &tria,
    KirasFM::ParameterReader prm,
    unsigned int layers_per_domin
  ) {

    // create the list of refrective indecies
    std::vector<double> n_list = {
      prm.get_refrective_index(0).real(),
      prm.get_refrective_index(1).real(),
      prm.get_refrective_index(2).real()
    };

    std::vector<double> index_list(intern_n_layers);
    for ( unsigned int i = 0; i < intern_n_layers; i++ ) {
      index_list[i] = n_list[intern_material_data_vector[i][0]];
    }

    std::vector<std::vector<double>> dz_list(N_domains);
    for ( unsigned int i = 0; i < N_domains; i++ ) {
      unsigned int start = i * layers_per_domin;
      unsigned int end   = ( (i + 1) * layers_per_domin > intern_n_layers ) ? intern_n_layers : ( (i + 1) * layers_per_domin > intern_n_layers );
      for ( unsigned int j = start; j < end; j++ )
        dz_list[i][j - start] = intern_material_data_vector[j][1];
    }

    help_make_thin_layer (
      tria,
      5,       /* number of cells in x-direction */
      5,       /* number of cells in y-direction */
      0.25,   /* cell size in x and y directions */
      dz_list,  /* cell size in x and y directions */
      index_list /* List of thickniss of each layer */
    );
  }


} // KirasFM_Grid_Generator


