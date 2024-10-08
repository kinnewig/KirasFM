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

#ifndef KIRASFM_GRID_GENERATOR_H
#define KIRASFM_GRID_GENERATOR_H

#include <fstream>
#include <vector>

// Distributed grid generator
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

// Grid generator
#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// KirasFM
#include <parameter_reader.h>

namespace KirasFM_Grid_Generator
{
  using namespace dealii;

  template <int dim>
  class KirasFMGridGenerator
  {
  public:
    // constructor
    KirasFMGridGenerator(const unsigned int domain_id,
                         const unsigned int N_domains,
                         const unsigned int refinements);

    // === Simple Waveguides ===
    void
    make_simple_waveguide(Triangulation<dim> &in);

    // === Y-beamsplitter ===
    void
    y_beamsplitter(Triangulation<dim> &tria);

    // === Diamond Fin ===
    void
    set_material_ids_for_diamond_fin(Triangulation<dim> &tria,
                                     const unsigned int  base_width_half = 5,
                                     const unsigned int  base_hight      = 3,
                                     const unsigned int  fin_hight       = 7,
                                     const unsigned int  depth           = 10,
                                     const double        block_size      = 1.0);

    void
    make_diamond_fin(Triangulation<dim> &tria,
                     const unsigned int  half_base_width,
                     const unsigned int  base_hight,
                     const unsigned int  fin_hight,
                     const unsigned int  buffer,
                     const unsigned int  depth);

    void
    refine_diamond_fin(Triangulation<dim> &tria,
                       const unsigned int  base_width_half,
                       const unsigned int  base_hight,
                       const unsigned int  fin_hight,
                       double              radius,
                       double              block_size);

    // === Nano Particle ===
    // --- Creation of the full embeddings ---
    void
    ball_embedding(Triangulation<dim> &tria,
                   double              outer_radius,
                   double              inner_radius,
                   bool                colorize = true);

    void
    ball_embedding_inverse(Triangulation<dim> &tria,
                           double              outer_radius,
                           bool                colorize = true);

    void
    shell_embedding(Triangulation<dim> &tria,
                    double              outer_radius,
                    double              inner_radius,
                    bool                colorize = true);

    // --- Creation of the half embeddings ---
    void
    ball_embedding_half(Triangulation<dim> &tria,
                        double              outer_radius,
                        double              inner_radius,
                        unsigned int        part,
                        bool                colorize = true);

    void
    ball_embedding_inverse_half(Triangulation<dim> &tria,
                                double              outer_radius,
                                unsigned int        part,
                                bool                colorize = true);

    void
    shell_embedding_half(Triangulation<dim> &tria,
                         double              outer_radius,
                         double              inner_radius,
                         unsigned int        part,
                         bool                colorize = true);

    // --- Creation of the eighth embeddings ---
    void
    ball_embedding_eighth(Triangulation<dim> &tria,
                          double              outer_radius,
                          double              inner_radius,
                          unsigned int        part,
                          bool                colorize = true);

    void
    ball_embedding_inverse_eighth(Triangulation<dim> &tria,
                                  double              outer_radius,
                                  unsigned int        part,
                                  bool                colorize = true);

    void
    shell_embedding_eighth(Triangulation<dim> &tria,
                           double              outer_radius,
                           double              inner_radius,
                           unsigned int        part,
                           bool                colorize = true);

    // --- create the nano particle itself ---
    void
    create_nano_particle(
      Triangulation<dim> &tria,
      double              ball_radius, /* radius of the silver ball */
      std::vector<double> /* List with the thickness of the dirrerent layers*/
    );

    void
    create_half_nano_particle(
      Triangulation<dim> &tria,
      double              ball_radius, /* radius of the silver ball */
      std::vector<double> /* List with the thickness of the dirrerent layers*/
    );


    void
    create_eighth_nano_particle(
      Triangulation<dim> &tria,
      double              ball_radius, /* radius of the silver ball */
      std::vector<double> /* List with the thickness of the dirrerent layers*/
    );

    // === Gallium Laser ===
    void
    make_gallium_laser(Triangulation<dim> &tria,
                       const unsigned int  outer_width,
                       const unsigned int  inner_width,
                       const unsigned int  hight,
                       const unsigned int  depth);

    void
    make_thin_layers(Triangulation<dim>      &tria,
                     KirasFM::ParameterReader prm,
                     unsigned int             layers_per_domin = 2);

    // === 6er Waveguide ===
    void
    make_6er_waveguide_parted(Triangulation<dim> &tria,
                              unsigned int        i,
                              const unsigned int  refinements,
                              const unsigned      last,
                              const double        scale);

    void
    make_6er_waveguide(Triangulation<dim> &tria,
                       unsigned int        i,
                       const unsigned int  refinements,
                       const unsigned      last,
                       const double        scale = 1.0);

  private:
    void
    help_make_thin_layer(
      Triangulation<dim> &tria,
      const unsigned int  steps_x,       /* number of cells in x-direction */
      const unsigned int  steps_y,       /* number of cells in y-direction */
      const double        block_size_xy, /* cell size in x and y directions */
      const std::vector<std::vector<double>>
        dz_list, /* cell size in x and y directions */
      const std::vector<double>
        material_list /* List of thickniss of each layer */
    );

    // internal functions
    void
    set_boundary_ids(Triangulation<dim> &in, const int axis);

    void
    refine_circular(
      Triangulation<dim> &in,
      Point<dim>          center,
      unsigned int        axis, /* 0 -> x-axis, 1 -> y-axis, 2 -> z-axis */
      double              radius);

    // Number of domains & domain_id
    const unsigned int domain_id;
    const unsigned int N_domains;
    const unsigned int refinements;
  };


} // namespace KirasFM_Grid_Generator

#endif
