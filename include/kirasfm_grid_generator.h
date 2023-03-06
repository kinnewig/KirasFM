#ifndef KIRASFM_GRID_GENERATOR_H
#define KIRASFM_GRID_GENERATOR_H

#include <fstream>
#include <vector>

// Distributed grid generator
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

// Grid generator
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/geometry_info.h>

// KirasFM
#include<parameter_reader.h>

namespace KirasFM_Grid_Generator {
  using namespace dealii;

  template <int dim> class KirasFMGridGenerator {
    public:
      // constructor
      KirasFMGridGenerator(
        const unsigned int domain_id,
        const unsigned int N_domains,
        const unsigned int refinements
      );

      // === Simple Waveguides ===
      void make_simple_waveguide( Triangulation<dim>& in );
      void make_embedded_ball( Triangulation<dim>& in );

      // === Y-beamsplitter ===
      void y_beamsplitter (
        Triangulation<dim>& tria
      );

      // === Diamond Fin ===
      void set_material_ids_for_diamond_fin(
        Triangulation<dim>& tria,
        const unsigned int base_width_half = 5,
        const unsigned int base_hight      = 3,
        const unsigned int fin_hight       = 7,
        const unsigned int depth           = 10,
        const double       block_size      = 1.0
      );

      void make_diamond_fin(
        Triangulation<dim>& tria,
        const unsigned int half_base_width,
        const unsigned int base_hight,
        const unsigned int fin_hight,
        const unsigned int buffer,
        const unsigned int depth
      );

      void refine_diamond_fin(
        Triangulation<dim>& tria,
        const unsigned int base_width_half,
        const unsigned int base_hight,
        const unsigned int fin_hight,
		double radius,
        double block_size
      );

      // === Nano Particle ===
      void
      quarter_ball_embedding (
        Triangulation<dim> &tria,
        double outer_radius, /* outer radius */
        double inner_radius  /* inner radius */
      );

      void
      quarter_ball_embedding_inverse (
              Triangulation<dim> &tria,
              double outer_radius, /* outer radius */
              double inner_radius  /* inner radius */
      );

      void
      ball_embedding (
        Triangulation<dim> &tria,
        double outer_radius, /* outer radius */
        double inner_radius  /* inner radius */
      );

      void
      shell_embedding (
              Triangulation<dim> &tria,
              double outer_radius, /* outer radius */
              double inner_radius  /* inner radius */
      );

      void
      shell_embedding_filled (
        Triangulation<dim> &tria,
        double outer_radius
      );

      void
      hyper_ball_embedded (
        Triangulation<dim> &tria,
        double ball_radius, /* radius of the silver ball */
        std::vector<double> /* List with the thickness of the dirrerent layers*/
      );



      // === Gallium Laser ===
      void
      make_gallium_laser (
        Triangulation<dim> &tria,
        const unsigned int outer_width,
        const unsigned int inner_width,
        const unsigned int hight,
        const unsigned int depth
      );

      void
      make_thin_layers (
        Triangulation<dim> &tria,
        KirasFM::ParameterReader prm,
        unsigned int layers_per_domin = 2
      );

      // === 6er Waveguide ===
      void 
      make_6er_waveguide_parted (
        Triangulation<dim> &tria,
        unsigned int i,
        const unsigned int refinements,
        const unsigned last,
        const double scale
      );

      void 
      make_6er_waveguide(
        Triangulation<dim> &tria,
        unsigned int i,
        const unsigned int refinements,
        const unsigned last,
        const double scale = 1.0
      ); 

    private:
      void
      help_make_thin_layer (
        Triangulation<dim> &tria,
        const unsigned int              steps_x,       /* number of cells in x-direction */
        const unsigned int              steps_y,       /* number of cells in y-direction */
        const double                    block_size_xy, /* cell size in x and y directions */
        const std::vector<std::vector<double>> dz_list,  /* cell size in x and y directions */
        const std::vector<double>       material_list  /* List of thickniss of each layer */
      );

      // internal functions
      void set_boundary_ids( Triangulation<dim>& in, const int axis);

      // Number of domains & domain_id
      const unsigned int domain_id;
      const unsigned int N_domains;
      const unsigned int refinements;

  };


} // namespace: DDM_Grid_Generator 

#endif
