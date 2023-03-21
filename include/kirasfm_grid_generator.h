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
      // eighth embeddings:
      void eighth_ball_embedding ( Triangulation<dim> &tria, const double inner_radius, const double outer_radius );
      void eighth_ball_embedding_inverse ( Triangulation<dim> &tria, const double inner_radius,  const double outer_radius );
      void eighth_shell_embedding ( Triangulation<dim> & tria, const double inner_radius, const double outer_radius );

      // quarter embeddings:
      void quarter_ball_embedding ( Triangulation<dim> &tria, const double inner_radius, const double outer_radius, const unsigned int part, bool colorize = true );
      void quarter_ball_embedding_inverse ( Triangulation<dim> &tria, const double outer_radius, const unsigned int part, const bool colorize = true );
      void quarter_shell_embedding ( Triangulation<dim> & tria, const double inner_radius, const double outer_radius, const unsigned int part, bool colorize = true );

      // quarter embeddings:
      void half_ball_embedding ( Triangulation<dim> &tria, const double inner_radius, const double outer_radius, const unsigned int part, bool colorize = true );
      void half_ball_embedding_inverse ( Triangulation<dim> &tria, const double outer_radius, const unsigned int part, const bool colorize = true );
      void half_shell_embedding ( Triangulation<dim> & tria, const double inner_radius, const double outer_radius, const unsigned int part, bool colorize = true );

      // full embeddings:
      void full_ball_embedding ( Triangulation<dim> &tria, const double inner_radius, const double outer_radius, bool colorize = true );
      void full_ball_embedding_inverse ( Triangulation<dim> &tria, const double outer_radius, const bool colorize = true );
      void full_shell_embedding ( Triangulation<dim> & tria, const double inner_radius, const double outer_radius, bool colorize = true );

      // create the nano particle itself
      void create_quarter_nano_particle ( Triangulation<dim> &tria, const double ball_radius, std::vector<double> layer_thickness );
      void create_half_nano_particle ( Triangulation<dim> &tria, const double ball_radius, std::vector<double> layer_thickness );
      void create_nano_particle ( Triangulation<dim> &tria, const double ball_radius, std::vector<double> layer_thickness );

      void
      refine_nano_particle (
        Triangulation<dim> &tria,
        double radius_refine,
        double radius_coarsen
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

      void refine_circular(
        Triangulation<dim>& in,
        Point<dim> center,
        unsigned int axis,    /* 0 -> x-axis, 1 -> y-axis, 2 -> z-axis */
        double radius
      );

      // Number of domains & domain_id
      const unsigned int domain_id;
      const unsigned int N_domains;
      const unsigned int refinements;
  };


} // namespace: DDM_Grid_Generator 

#endif
