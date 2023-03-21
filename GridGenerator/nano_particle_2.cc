//
// Created by sebastian on 19.03.23.
//

#include <kirasfm_grid_generator.h>

namespace KirasFM_Grid_Generator {
  using namespace dealii;


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

  // Help Function: Part Creator
  // Colorize: x: 0, 1
  //           y: 2, 3
  //           z: 4, 5
  // Inner ball : 6
  // outer ball : 7
  template <int dim>
  void part_creator (
          Triangulation<dim> &tria,
          const double        inner_radius,  /* inner radius */
          const double        outer_radius,  /* outer radius */
          const unsigned int  part
  ) {
    const double pi = 3.1415926535897932384626433832795028841971693993751058209749445923;

    // check that we stay in range
    AssertIndexRange(part, 8);

    if (part > 3)
      GridTools::rotate(
        Tensor<1, 3, double>({0.0, 0.0, 1.0}),
        pi / 2.0,
        tria
      );

    GridTools::rotate(
      Tensor<1, 3, double>({1.0, 0.0, 0.0}),
      ( (part % 4) * pi) / 2.0,
      tria
    );

    double TOL = 0.001;

    int x_front = ( part < 4 ) ? 0 : 1;
    int x_back  = ( part < 4 ) ? 1 : 0;
    int y_front = ( part % 4 == 0 || part % 4 == 3 ) ? 2 : 3;
    int y_back  = ( part % 4 == 0 || part % 4 == 3 ) ? 3 : 2;
    int z_front = ( part % 4 == 0 || part % 4 == 1 ) ? 4 : 5;
    int z_back  = ( part % 4 == 0 || part % 4 == 1 ) ? 5 : 4;

    // colorize
    for ( const auto &cell : tria.cell_iterators() )
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
        //skip all faces, that are not located at the boundary
        if ( !cell->face(face)->at_boundary() )
          continue;

        // x direction
        if ( std::abs(cell->face(face)->center()[0]) < TOL )
          cell->face(face)->set_boundary_id(x_front);

        else if ( std::abs(cell->face(face)->center()[1]) < TOL )
          cell->face(face)->set_boundary_id(y_front);

        else if ( std::abs(cell->face(face)->center()[2]) < TOL )
          cell->face(face)->set_boundary_id(z_front);

        else if ( std::abs(std::abs(cell->face(face)->center()[0]) - outer_radius) < TOL )
          cell->face(face)->set_boundary_id(x_back);

        else if ( std::abs(std::abs(cell->face(face)->center()[1]) - outer_radius) < TOL )
          cell->face(face)->set_boundary_id(y_back);

        else if ( std::abs(std::abs(cell->face(face)->center()[2]) - outer_radius) < TOL )
          cell->face(face)->set_boundary_id(z_back);

        else if( cell->face(face)->center().norm() < (inner_radius + outer_radius) / 2) {
          cell->face(face)->set_all_manifold_ids(0);
          cell->face(face)->set_boundary_id(6);
        }

        else if( cell->face(face)->center().norm() > (inner_radius + outer_radius) / 2) {
          cell->face(face)->set_all_manifold_ids(0);
          cell->face(face)->set_boundary_id(7);
        }

      }

    tria.set_manifold(0, SphericalManifold<dim>(Point<dim>(0,0,0)));
  }

    template<int dim>
    void mark_layer_intern_interface (
            Triangulation<dim> &tria,
            const unsigned int  domain_id
    ) {
      const unsigned int layer_id      = domain_id / 8;
      const unsigned int subdomain_id  = domain_id % 8;

      const unsigned int neighbor_case = (subdomain_id < 4) ? subdomain_id : subdomain_id - 4;
      double neighbor[4][2]            = {{2, 4}, {3, 4}, {5, 3}, {5, 2}};
      double neighbor_shift[4][2]      = {{1,3}, {-1,1}, {-1,1}, {-3,-1}};

      const unsigned int side       = (subdomain_id < 4) ? 0 :  1;
      const int          side_shift = (subdomain_id < 4) ? 4 : -4;

      for ( const auto &cell : tria.cell_iterators() )
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          // skip all faces that are not located at the boundary
          if ( !cell->face(face)->at_boundary() )
            continue;

          if ( cell->face(face)->boundary_id() == neighbor[neighbor_case][0] )
            cell->face(face)->set_boundary_id(domain_id + neighbor_shift[neighbor_case][0] + 2);

          else if ( cell->face(face)->boundary_id() == neighbor[neighbor_case][1] )
            cell->face(face)->set_boundary_id(domain_id + neighbor_shift[neighbor_case][1] + 2);

          else if ( cell->face(face)->boundary_id() == side )
            cell->face(face)->set_boundary_id(domain_id + side_shift + 2);

          else if ( cell->face(face)->boundary_id() == 6 )
            cell->face(face)->set_boundary_id(domain_id + 8 + 2);

          else if ( cell->face(face)->boundary_id() == 7 )
            cell->face(face)->set_boundary_id(domain_id - 8 + 2);

        }
    }

  template<int dim>
  void mark_layer_intern_interface_special (
          Triangulation<dim> &tria,
          const unsigned int  domain_id
  ) {
    const unsigned int layer_id      = domain_id / 8;
    const unsigned int subdomain_id  = domain_id % 8;

    const unsigned int neighbor_case = (subdomain_id < 4) ? subdomain_id : subdomain_id - 4;
    double neighbor[4][2]            = {{2, 4}, {3, 4}, {5, 3}, {5, 2}};
    double neighbor_shift[4][2]      = {{1,3}, {-1,1}, {-1,1}, {-3,-1}};

    const unsigned int side       = (subdomain_id < 4) ? 0 :  1;
    const int          side_shift = (subdomain_id < 4) ? 4 : -4;
    double outsides[8][3] = {
            {1, 3, 5},
            {1, 2, 5},
            {1,2,4},
            {1,3,4},
            {0, 3, 5},
            {0, 2, 5},
            {0, 2, 4},
            {0,3,4}
    };

    const unsigned int bc_dirichlet = 32 + 2;
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

    for ( const auto &cell : tria.cell_iterators() )
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
        // skip all faces that are not located at the boundary
        if ( !cell->face(face)->at_boundary() )
          continue;

        if ( cell->face(face)->boundary_id() == neighbor[neighbor_case][0] )
          cell->face(face)->set_boundary_id(domain_id + neighbor_shift[neighbor_case][0] + 2);

        else if ( cell->face(face)->boundary_id() == neighbor[neighbor_case][1] )
          cell->face(face)->set_boundary_id(domain_id + neighbor_shift[neighbor_case][1] + 2);

        else if ( cell->face(face)->boundary_id() == side )
          cell->face(face)->set_boundary_id(domain_id + side_shift + 2);

        else if ( cell->face(face)->boundary_id() == 6 )
          cell->face(face)->set_boundary_id(domain_id + 8 + 2);

        else if ( cell->face(face)->boundary_id() == 7 )
          cell->face(face)->set_boundary_id(domain_id - 8 + 2);

        else if ( cell->face(face)->boundary_id() == outsides[subdomain_id][0] ) {
          cell->face(face)->set_boundary_id(outsides_associated_ids[subdomain_id][0]);
          if (outsides_associated_ids[subdomain_id][2] == bc_robin)
            cell->set_material_id(0);
        }

        else if ( cell->face(face)->boundary_id() == outsides[subdomain_id][1] ) {
          cell->face(face)->set_boundary_id(outsides_associated_ids[subdomain_id][1]);
          if (outsides_associated_ids[subdomain_id][2] == bc_robin)
            cell->set_material_id(0);
        }

        else if ( cell->face(face)->boundary_id() == outsides[subdomain_id][2] ) {
          cell->face(face)->set_boundary_id(outsides_associated_ids[subdomain_id][2]);
          if (outsides_associated_ids[subdomain_id][2] == bc_robin)
            cell->set_material_id(0);
        }
      }
  }

  template<int dim>
  void KirasFMGridGenerator<dim>::ball_embedding (
    Triangulation<dim> &tria,
    double outer_radius,
    double inner_radius,
    const unsigned int part
  ) {
    // create the quarter ball embedding
    quarter_ball_embedding ( tria, outer_radius, inner_radius );

    // rotate it into the correct orientation and colorize it
    part_creator ( tria, inner_radius, outer_radius, part );

    tria.refine_global(refinements);
  }

  template<int dim>
  void KirasFMGridGenerator<dim>::ball_embedding_inverse(
    Triangulation<dim> &tria,
    double outer_radius,
    const unsigned int part
  ) {

    // create the inverse quarter ball embedding
    const double inner_radius = 0.3 * outer_radius;
    quarter_ball_embedding_inverse (
      tria,
      outer_radius,
      inner_radius
    );

    // Now fill the center
    Triangulation<dim> center_tria;
    GridGenerator::hyper_rectangle(
            center_tria,
            Point<dim>(0,0,0),
            Point<dim>(inner_radius,inner_radius,inner_radius)
    );
    GridGenerator::merge_triangulations(tria, center_tria, tria, 0.1 * inner_radius);

    // rotate it into the correct orientation and colorize it
    part_creator ( tria, inner_radius, outer_radius, part );

    tria.refine_global(refinements);
  }

  template<int dim>
  void KirasFMGridGenerator<dim>::shell_embedding(
    Triangulation<dim> &tria,
    double outer_radius,
    double inner_radius,
    const unsigned int part
  ) {
    // create the quarter shell
    quarter_shell_embedding (
      tria,
      outer_radius,
      inner_radius
    );

    // rotate it into the correct orientation and colorize it
    part_creator ( tria, inner_radius, outer_radius, part );

    tria.refine_global(refinements);
  }

  template<int dim>
  void KirasFMGridGenerator<dim>::create_parted_nano_particle (
    Triangulation<dim> &tria,
    double ball_radius,
    std::vector<double> layer_thickness
  ) {
    // This function will be nasty, first of all. each layer will consit out of 8 subdomains.
    const unsigned int layer_id     = domain_id / 8;
    const unsigned int subdomain_id = domain_id % 8;

    // check that we stay in range
    AssertIndexRange(layer_id, layer_thickness.size() + 1);

    // Create the most outer shell
    if ( layer_id == 0 ) {
      double inner_radius = layer_thickness[layer_id + 1];
      double outer_radius = layer_thickness[layer_id];

      ball_embedding(tria, outer_radius, inner_radius, subdomain_id);

      // mark interfaces
      mark_layer_intern_interface_special(tria, domain_id);
    }

    // create the center
    else if ( layer_id == layer_thickness.size() - 1 ) {
      double outer_radius = layer_thickness[layer_id];

      ball_embedding_inverse(tria, outer_radius, subdomain_id);

      // mark interfaces
      mark_layer_intern_interface(tria, domain_id);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);
    }

    else if ( layer_id == layer_thickness.size() ) {
      double outer_radius = layer_thickness[0];

      double TOL                = 0.001;
      double thickness          = 0.8;
      unsigned int bc_dirichlet = 1;

      // create the grid
      unsigned int r = std::pow(2, refinements + 1);
      GridGenerator::subdivided_hyper_rectangle (
        tria,
        std::vector<unsigned int>({r, r, refinements}),
        Point<dim>(-outer_radius, -outer_radius, -outer_radius),
        Point<dim>(outer_radius, outer_radius, -outer_radius - thickness)
      );

      // mark the interfaces
      for ( const auto &cell : tria.cell_iterators() )
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++) {
          // skip all faces that are not located at the boundary
          if ( !cell->face(face)->at_boundary() )
            continue;

          if ( std::abs(cell->face(face)->center()[2] + outer_radius) < TOL ) {
            if ( cell->face(face)->center()[0] > 0 ) {
              if ( cell->face(face)->center()[1] > 0 ) {
                cell->face(face)->set_boundary_id(3 + 2);
              } else {
                cell->face(face)->set_boundary_id(2 + 2);
              }
            } else {
              if ( cell->face(face)->center()[1] > 0 ) {
                cell->face(face)->set_boundary_id(7 + 2);
              } else {
                cell->face(face)->set_boundary_id(6 + 2);
              }
            }
          }

          if ( std::abs(cell->face(face)->center()[2] + (outer_radius + thickness)) < TOL ) {
            cell->face(face)->set_boundary_id(bc_dirichlet);
          }

        }
    }

    else {
      double inner_radius = layer_thickness[layer_id + 1];
      double outer_radius = layer_thickness[layer_id];

      shell_embedding(tria, outer_radius, inner_radius, subdomain_id);

      // mark interfaces
      mark_layer_intern_interface(tria, domain_id);

      // mark a ball in the center with a different material id
      mark_ball<dim>(tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1);
    }

  }

  template class KirasFMGridGenerator<3>;
} // namespace: KirasFM_Grid_Generator