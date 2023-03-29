#ifndef HANGING_NODES_H
#define HANGING_NODES_H

// === C++ includes ===
#include <iostream>

// === deal.II includes ==
// constraints
#include <deal.II/lac/affine_constraints.h>

// dof handler
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

namespace KirasFM {
using namespace dealii;

template <int dim>
void bullshit_hanging_nodes_constraints(
  const DoFHandler<dim>     &dof_handler,
  AffineConstraints<double> &hanging_node_constraints,
  bool                      verbose = false
) {

  Assert (dim == 2, ExcNotImplemented());

  std::vector<types::global_dof_index> face_dof_indices;

  std::map<types::global_dof_index,std::set<types::global_dof_index> > depends_on;
  
  // loop over all cells
  for ( const auto &cell : dof_handler.active_cell_iterators() ) {

    //artificial cells can at best neighbor ghost cells, but we're not
    // interested in these interfaces
    if ( cell->is_artificial() )
      continue;
    
    // loop over all faces:
    for ( unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f ) {

      // check of the neighbor is refined; if so, we need to
      // treat the constraints on this interface
      if (cell->face(f)->has_children() == false)
        continue;

      // get fe
      const FiniteElement<dim> &fe = cell->get_fe();

      const unsigned int n_dofs = fe.n_dofs_per_line();
      face_dof_indices.resize(n_dofs);

      cell->face(f)->get_dof_indices( face_dof_indices );
      const std::vector<types::global_dof_index> dof_on_mother_face = face_dof_indices;
      
      cell->face(f)->child(0)->get_dof_indices (face_dof_indices);
      const std::vector<types::global_dof_index> dof_on_child_face_0 = face_dof_indices;

      cell->face(f)->child(1)->get_dof_indices (face_dof_indices);
      const std::vector<types::global_dof_index> dof_on_child_face_1 = face_dof_indices;

      // get the direction of the faces
      const int direction_mother = 
        (cell->face(f)->vertex_index(0) > cell->face(f)->vertex_index(1)) ? 1 : -1;
      const int direction_child_0 = 
        (cell->face(f)->child(0)->vertex_index(0) > cell->face(f)->child(0)->vertex_index(1)) ? 1 : -1;
      const int direction_child_1 = 
        (cell->face(f)->child(1)->vertex_index(0) > cell->face(f)->child(1)->vertex_index(1)) ? 1 : -1;

      // now enter a constraint that child face dofs are
      // linear combinations of the parent face dofs
      for( unsigned int row = 0; row < n_dofs; row++ ) {
        hanging_node_constraints.add_line (dof_on_child_face_0[row]);
        hanging_node_constraints.add_line (dof_on_child_face_1[row]);
      }

      for( unsigned int row = 0; row < n_dofs; row++ ) {
        for( unsigned int dof_i_on_mother = 0; dof_i_on_mother < n_dofs; dof_i_on_mother++ ) {
          unsigned int shift_0 = ( direction_mother == direction_child_0 ) ? 0 : n_dofs;
          hanging_node_constraints.add_entry (
            dof_on_child_face_0[row],
            dof_on_mother_face[dof_i_on_mother],
            fe.constraints()(row + shift_0, dof_i_on_mother)
          );

          unsigned int shift_1 = ( direction_mother == direction_child_1 ) ? 0 : n_dofs;
          hanging_node_constraints.add_entry (
            dof_on_child_face_1[row],
            dof_on_mother_face[dof_i_on_mother],
            fe.constraints()(row + shift_1, dof_i_on_mother)
          );
        }
      }

      if ( verbose ) {
        std::cout << "\tDoF on mother face: (" 
                  << dof_on_mother_face[0] << ", " 
                  << dof_on_mother_face[1] << "); " 
                  << std::endl;

        std::cout << "\tDoF on 1st child face: (" 
                  << dof_on_child_face_0[0] << ", " 
                  << dof_on_child_face_0[1] << "); " 
                  << std::endl;

        std::cout << "\tDoF on 2nd child face: (" 
                  << dof_on_child_face_1[0] << ", " 
                  << dof_on_child_face_1[1] << "); " 
                  << std::endl;

        std::cout << "\tdirections: mother = " << direction_mother 
                  << ", child_0 = " << direction_child_0 
                  << ", child_1 = " << direction_child_1 
                  << std::endl;
      }

    } // rof: faces 

  } // rof: cells

}

template <int dim>
void bullshit_hanging_nodes_constraints_3d(
  const DoFHandler<dim> &dof_handler,
  AffineConstraints<double> &hanging_node_constraints
) {

  // TODO: Remove the following two lines, only used for debugging
  bool printed = true;

  std::vector<types::global_dof_index> dofs_on_mother;
  std::vector<types::global_dof_index> dofs_on_children;
  
  // loop over all quads; only on quads there can be constraints. We do so
  // by looping over all active cells and checking whether any of the faces
  // are refined which can only be from the neighboring cell because this
  // one is active. In that case, the face is subject to constraints
  //
  // note that even though we may visit a face twice if the neighboring
  // cells are equally refined, we can only visit each face with hanging
  // nodes once
  for ( const auto &cell : dof_handler.active_cell_iterators() ) {

    // Skip artificial cells:
    // artificial cells can at best neighbor ghost cells, but we're not
    // interested in these interfaces
    if ( cell->is_artificial() )
      continue;

    // Loop over all faces
    for ( const unsigned int face : cell->face_indices() ) {

      // Skip cells without children
      if ( cell->face(face)->has_children() == false )
        continue;

      // first of all, make sure that we treat a case which is
      // possible, i.e. either no dofs on the face at all or no
      // anisotropic refinement
      if (cell->get_fe().n_dofs_per_face(face) == 0)
        continue;

      Assert(cell->face(face)->refinement_case() ==
         RefinementCase<dim - 1>::isotropic_refinement,
       ExcNotImplemented());

      // in any case, faces can have at most two active FE indices,
      // but here the face can have only one (namely the same as that
      // from the cell we're sitting on), and each of the children can
      // have only one as well. check this
      AssertDimension(cell->face(face)->n_active_fe_indices(), 1);
      Assert(cell->face(face)->fe_index_is_active(
               cell->active_fe_index()) == true,
             ExcInternalError());
      for (unsigned int c = 0; c < cell->face(face)->n_children();
           ++c)
        if (!cell->neighbor_child_on_subface(face, c)
               ->is_artificial())
          AssertDimension(
            cell->face(face)->child(c)->n_active_fe_indices(), 1);


      // right now, all that is implemented is the case that both
      // sides use the same fe, and not only that but also that all
      // lines bounding this face and the children have the same FE
      for (unsigned int c = 0; c < cell->face(face)->n_children();
           ++c)
        if (!cell->neighbor_child_on_subface(face, c)
               ->is_artificial())
          {
            Assert(cell->face(face)->child(c)->fe_index_is_active(
                     cell->active_fe_index()) == true,
                   ExcNotImplemented());
            for (unsigned int e = 0; e < GeometryInfo<dim>::vertices_per_face; ++e)
              {
                Assert(cell->face(face)
                           ->child(c)
                           ->line(e)
                           ->n_active_fe_indices() == 1,
                       ExcNotImplemented());
                Assert(cell->face(face)
                           ->child(c)
                           ->line(e)
                           ->fe_index_is_active(
                             cell->active_fe_index()) == true,
                       ExcNotImplemented());
              }
          }
      for (unsigned int e = 0; e < GeometryInfo<dim>::vertices_per_face; ++e)
        {
          Assert(cell->face(face)->line(e)->n_active_fe_indices() ==
                   1,
                 ExcNotImplemented());
          Assert(cell->face(face)->line(e)->fe_index_is_active(
                   cell->active_fe_index()) == true,
                 ExcNotImplemented());
        }

      // Ok, start up the work
      // Get the FE
      const FiniteElement<dim> &fe       = cell->get_fe();
      const unsigned int        fe_index = cell->active_fe_index();

      // get the polynomial degree
      unsigned int degree(fe.degree);

      // Get the number of DoFs on mother and children
      // Number of DoFs on the mother
      const unsigned int n_dofs_on_mother = fe.n_dofs_per_face(face);
      dofs_on_mother.resize(n_dofs_on_mother);

      // Number of intern lines of the children
      // for more details see: Decription of the class GeometryInfo<dim>
      // .................
      // .       |       .
      // .  c2   1   c3  .
      // .       |       .
      // .---2---+---3---.
      // .       |       .
      // .  c0   0   c1  .
      // .       |       .
      // .................
      const unsigned int n_intern_lines_on_children = 4;

      // Number of external lines of the children
      // +---6--------7--+
      // |       .       |
      // 1  c2   .   c3  3
      // |       .       |
      // | . . . . . . . |
      // |       .       |
      // 0  c0   .   c1  2
      // |       .       |
      // +---4---+---5---+
      const unsigned int n_external_lines_on_children = 8;

      const unsigned int n_lines_on_children =
    		  n_intern_lines_on_children + n_external_lines_on_children;

      const unsigned int n_of_children = 4;

      // Number of DoFs on the children
      // Remark: Nedelec elements have no DoFs on the vertices, therefore we skip the vertices
      const unsigned int n_dofs_on_children = ( n_lines_on_children * fe.n_dofs_per_line()
    		  + n_of_children * fe.n_dofs_per_quad(face) );
      // we might not use all of those in case of artificial cells, so
      // do not resize(), but reserve() and use push_back later.
      dofs_on_children.clear();
      dofs_on_children.reserve(n_dofs_on_children);

      Assert(n_dofs_on_mother == fe.constraints().n(),
             ExcDimensionMismatch(n_dofs_on_mother,
                                  fe.constraints().n()));
      Assert(n_dofs_on_children == fe.constraints().m(),
             ExcDimensionMismatch(n_dofs_on_children,
                                  fe.constraints().m()));

      // Get the current face
      const typename DoFHandler<3, 3>::face_iterator this_face = cell->face(face);

      // --------------------------------------------------------------------------------------------------
      // Fill the DoFs on the mother:
      unsigned int next_index = 0;

      // DoFs on vertices
      // Nedelec elements have no DoFs on the vertices

      // DoFs on lines
      for ( unsigned int line = 0; line < GeometryInfo<dim>::lines_per_face; ++line )
        for ( unsigned int dof = 0; dof != fe.n_dofs_per_line(); ++dof )
          dofs_on_mother[next_index++] = this_face->line(line)->dof_index(dof, fe_index);

      // DoFs on the face
      for ( unsigned int dof = 0; dof != fe.n_dofs_per_quad(face); ++dof )
        dofs_on_mother[next_index++] = this_face->dof_index(dof, fe_index);

      // Check that we have added all DoFs
      AssertDimension(next_index, dofs_on_mother.size());

      // Fill the DoF on the children::
      Assert(dof_handler.get_triangulation()
                 .get_anisotropic_refinement_flag() ||
               ((this_face->child(0)->vertex_index(3) ==
                 this_face->child(1)->vertex_index(2)) &&
                (this_face->child(0)->vertex_index(3) ==
                 this_face->child(2)->vertex_index(1)) &&
                (this_face->child(0)->vertex_index(3) ==
                 this_face->child(3)->vertex_index(0))),
             ExcInternalError());

      // DoFs on vertices
      // Nedelec elements have no DoFs on the vertices

      // DoFs on lines
      // next the DoFs on the interior lines to the children; the order of
      // these lines is shown above (see n_intern_lines_on_children)
      for ( unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof )
        dofs_on_children.push_back( this_face->child(0)->line(1)->dof_index(dof, fe_index) );

      for (unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof)
        dofs_on_children.push_back( this_face->child(2)->line(1)->dof_index(dof, fe_index) );

      for (unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof)
        dofs_on_children.push_back( this_face->child(0)->line(3)->dof_index(dof, fe_index) );

      for (unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof)
        dofs_on_children.push_back( this_face->child(1)->line(3)->dof_index(dof, fe_index) );

      // DoFs on the bordering lines
      // DoFs on the exterior lines to the children; the order of
      // these lines is shown above (see n_external_lines_on_children)
      for ( unsigned int line = 0; line < 4; ++line )
        for ( unsigned int child = 0; child < 2; ++child )
          for (unsigned int dof = 0; dof != fe.n_dofs_per_line(); ++dof )
            dofs_on_children.push_back( this_face->line(line)->child(child)->dof_index(dof, fe_index) );

      // DoFs on the faces of the four children
        for ( unsigned int child = 0; child < n_of_children; ++child ) {
          // skip artificial cells
          if ( cell->neighbor_child_on_subface(face, child)->is_artificial() )
            continue;

          for ( unsigned int dof = 0; dof != fe.n_dofs_per_quad(face); ++dof )
            dofs_on_children.push_back( this_face->child(child)->dof_index(dof, fe_index) );
        } // rof: child

        // consitency check:
        // note: can get fewer DoFs when we have artificial cells:
        Assert(dofs_on_children.size() <= n_dofs_on_children,
          ExcInternalError());

      // --------------------------------------------------------------------------------------------------
      // As the Nedelec elements are oriented we need to take care of the orientation of
      // the lines.

      // Remark: false means the line is not flipped. True means the line is flipped

      // Orientation - Lines:
      // Get the orientation from the edges from the mother cell
      std::vector<bool> direction_mother(GeometryInfo<dim>::lines_per_face, false);
      for ( unsigned int line = 0; line < GeometryInfo<dim>::lines_per_face; ++line )
        if( this_face->line(line)->vertex_index(0) > this_face->line(line)->vertex_index(1) )
          direction_mother[line] = true;

      // Get the orientation from the intern edges of the children
      std::vector<bool> direction_child_intern(n_intern_lines_on_children, false);
      // get the global label of center vertex
      unsigned int center = this_face->child(0)->vertex_index(3);
      // dof numbers on the centers of the lines bounding this face
      for ( unsigned int line = 0; line < n_intern_lines_on_children; ++line )
        if ( line % 2 == 0 ) {
          direction_child_intern[line] = this_face->line(line)->child(0)->vertex_index(1) < center ? false : true;
        }
        else {
          direction_child_intern[line] = this_face->line(line)->child(0)->vertex_index(1) > center ? false : true;
        }

      // Get the orienatiation the outer edges of the children
      std::vector<bool> direction_child(n_external_lines_on_children, false);
      for ( unsigned int line = 0; line < GeometryInfo<dim>::lines_per_face; ++line ) {
        if( this_face->line(line)->child(0)->vertex_index(0) > this_face->line(line)->child(0)->vertex_index(1) )
          direction_child[2 * line] = true;
        if( this_face->line(line)->child(1)->vertex_index(0) > this_face->line(line)->child(1)->vertex_index(1) )
          direction_child[2 * line + 1] = true;
      }


      // --------------------------------------------------------------------------------------------------
      // Orientation - Faces:
      bool mother_flip_x  = false;
      bool mother_flip_y  = false;
      bool mother_flip_xy = false;
      std::vector<bool> child_flip_x(n_of_children, false);
      std::vector<bool> child_flip_y(n_of_children, false);
      std::vector<bool> child_flip_xy(n_of_children, false);
      const unsigned int vertices_adjacent_on_face[GeometryInfo<3>::vertices_per_face][2] = { {1, 2}, {0, 3}, {3, 0}, {2, 1} };

      { // Mother
      // get thet position of the vertex with the highest number
        unsigned int current_glob = cell->face(face)->vertex_index(0);
        unsigned int current_max  = 0;
        for (unsigned int v = 1; v < GeometryInfo<dim>::vertices_per_face; ++v)
          if ( current_glob < this_face->vertex_index(v) ) {
            current_max = v;
            current_glob = this_face->vertex_index(v);
          }

        // if the vertex with the highest DoF index is in the lower row of the face,
        // the face is flipped in y direction
        if ( current_max < 2 )
          mother_flip_y = true;

        // if the vertex with the highest DoF index is on the left side of the face
        // the face is flipped in x direction
        if ( current_max % 2 == 0 )
          mother_flip_x = true;

        // Get the minor direction of the face of the mother
        if ( this_face->vertex_index(vertices_adjacent_on_face[current_max][0]) <
             this_face->vertex_index(vertices_adjacent_on_face[current_max][1])
           )
          mother_flip_xy = true;

      }

      // Children
      // Get the orientation of the faces of the children
      for ( unsigned int child = 0; child < n_of_children; ++child ) {
        unsigned int current_max  = 0;
        unsigned int current_glob = this_face->child(child)->vertex_index(0);

        for (unsigned int v = 1; v < GeometryInfo<dim>::vertices_per_face; ++v)
          if ( current_glob < this_face->child(child)->vertex_index(v) ) {
            current_max = v;
            current_glob = this_face->child(child)->vertex_index(v);
          }

        if ( current_max < 2 )
          child_flip_y[child] = true;

        if ( current_max % 2 == 0 )
          child_flip_x[child] = true;

        if ( this_face->child(child)->vertex_index(vertices_adjacent_on_face[current_max][0]) <
             this_face->child(child)->vertex_index(vertices_adjacent_on_face[current_max][1])
           )
          child_flip_xy[child] = true;

        child_flip_xy[child] = mother_flip_xy;
      }


      // Copy the constraint matrix, since we need to modify that matrix

      std::vector<std::vector<double>>
        constraints(n_lines_on_children * fe.n_dofs_per_line()
        		+ n_of_children * fe.n_dofs_per_quad(), std::vector<double>(dofs_on_mother.size(), 0));
          
      { // copy the constraint matrix

        for ( unsigned int line = 0; line < 4; line++ ) {
          unsigned int row_start  = line * fe.n_dofs_per_line();
          unsigned int line_mother = line / 2;
          unsigned int row_mother = (line_mother * 2) * fe.n_dofs_per_line();
          for ( unsigned int row = 0; row < fe.n_dofs_per_line(); row++ )
            for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_line(); i++ )
              constraints[row + row_start][i] = fe.constraints()(row + row_mother, i);
        }

        for ( unsigned int line = 0; line < 4; line++ ) {
          unsigned int row_start = line * fe.n_dofs_per_line();
          unsigned int line_mother = line / 2;
          unsigned int row_mother = (line_mother * 2) * fe.n_dofs_per_line();
          for ( unsigned int row = 0; row < fe.n_dofs_per_line(); row++ ){
            for ( unsigned int i = 4 * fe.n_dofs_per_line(); i <  dofs_on_mother.size(); i++ )
              constraints[row + row_start][i] = fe.constraints()(row + row_mother, i);
          }
        }

        // copy the weights for exterior edges
        unsigned int row_offset = 4 * fe.n_dofs_per_line();
        for ( unsigned int line = 0; line < 8; line++ ) {
          unsigned int row_start = line * fe.n_dofs_per_line();
          unsigned int line_mother = line / 2;
          unsigned int row_mother = (line_mother * 2) * fe.n_dofs_per_line();
          for ( unsigned int row = row_offset; row < row_offset + fe.n_dofs_per_line(); row++ )
            for ( unsigned int i = 0; i < dofs_on_mother.size(); i++ )
              constraints[row + row_start][i] = fe.constraints()(row + row_mother, i);
        }

        // copy the weights for the faces
        row_offset = 12 * fe.n_dofs_per_line();
        for ( unsigned int face = 0; face < 4; face++ ) {
          unsigned int row_start = face * fe.n_dofs_per_quad();
          for ( unsigned int row = row_offset; row < row_offset + fe.n_dofs_per_quad(); row++ )
            for ( unsigned int i = 0; i < dofs_on_mother.size(); i++ )
              constraints[row + row_start][i] = fe.constraints()(row, i);
        }
      }

        // Modify the matrix
          // Edge - Edge:
          //
          // Interior edges: the interior edges have support on the corresponding edges and faces
          // loop over all 4 intern edges
          for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_line(); i++ ) {
            unsigned int line_i = i / fe.n_dofs_per_line();
            unsigned int tmp_i  = i % degree;

            // loop over the edges of the mother cell
            for ( unsigned int j = 0; j < 4 * fe.n_dofs_per_line(); j++ ) {
              unsigned int line_j = j / fe.n_dofs_per_line();
              unsigned int tmp_j  = j % degree;

                if ( (line_i < 2 && line_j < 2) || (line_i >=2 && line_j >= 2) ) {
                  if ( direction_child_intern[line_i] != direction_mother[line_j] )
                    if ( ( tmp_i + tmp_j ) % 2 == 0 ) { // symmetric
                      constraints[i][j] *= 1.0;
                    }
                    else { // anti-symmetric
                      constraints[i][j] *= -1.0;
                    }
                }
                else {
                  if ( direction_mother[line_i] ) // TODO: It is pure random that I found this relation, think about why this holds true!
                    if ( ( tmp_i + tmp_j ) % 2 == 0 ) { // symmetric
                      constraints[i][j] *= 1.0;
                    }
                    else { // anti-symmetric
                      constraints[i][j] *= -1.0;
                    }
                }
            } // rof: DoF j

          }  // rof: DoF i

          // Exterior edges
          for ( unsigned int i = 4 * fe.n_dofs_per_line(); i < 12 * fe.n_dofs_per_line(); i++ ) {
            unsigned int line_i = ( i / fe.n_dofs_per_line() ) - 4;
            unsigned int tmp_i = i % degree;

            // loop over the edges of the mother cell
            for ( unsigned int j = 0; j < 4 * fe.n_dofs_per_line(); j++ ) {
              unsigned int line_j = j / fe.n_dofs_per_line();
              unsigned int tmp_j = j % degree;

              if ( direction_child[line_i] != direction_mother[line_j] )
                if ( ( tmp_i + tmp_j ) % 2 == 0 ) { // symmetric
                  constraints[i][j] *= 1.0;
                }
                else { // anti-symmetric
                  constraints[i][j] *= -1.0;
                }

            } // rof: DoF j
          }  // rof: DoF i

          // Edge - Face
          //
          // Interior edges: // TODO: for y
          for ( unsigned int i = 0; i < 2 * fe.n_dofs_per_line(); i++ ) {
            unsigned int line_i = i / fe.n_dofs_per_line();
            unsigned int tmp_i  = i % degree;

            unsigned int start_j = 4 * fe.n_dofs_per_line();

            for ( unsigned int block = 0; block < 2; block++ ) {
              // type 1:
              for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {
                  unsigned int j = start_j + jx + ( jy * (degree - 1) );
                  if ( direction_child_intern[line_i] != mother_flip_y )
                    if ( ( jy + tmp_i ) % 2 == 0 ) { // anti-symmetric case
                      constraints[i][j] *= -1.0;
                    }
                    else { // symmetric case
                      constraints[i][j] *= 1.0;
                    }

                } // rof : jy

              start_j += (degree - 1) * (degree - 1);

              // type 2:
              for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {
                  unsigned int j = start_j + jx + ( jy * (degree - 1) );

                  if ( direction_child_intern[line_i] != mother_flip_y )
                    if ( ( jy + tmp_i ) % 2 == 0 ) { // anti-symmetric case
                      constraints[i][j] *= -1.0;
                    }
                    else { // symmetric case
                      constraints[i][j] *= 1.0;
                    }

                }
              start_j += (degree - 1) * (degree - 1);

              // type 3.1:
                // TODO: skip
                start_j += degree - 1;

              // type 3.2:
                // TODO: skip
                start_j += degree - 1;

            } // rof: block
          } // rof: DoF i

          // Interior edges: // TODO: for x
          for ( unsigned int i = 2 * fe.n_dofs_per_line(); i < 4 * fe.n_dofs_per_line(); i++ ) {
            unsigned int line_i = i / fe.n_dofs_per_line();
            unsigned int tmp_i  = i % degree;

            unsigned int start_j = 4 * fe.n_dofs_per_line();

            for ( unsigned int block = 0; block < 2; block++ ) {
              // type 1:
              for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {
                  unsigned int j = start_j + jx + ( jy * (degree - 1) );
                  if ( direction_child_intern[line_i] != mother_flip_x )
                    if ( ( jx + tmp_i ) % 2 == 0 ) { // anti-symmetric case
                      constraints[i][j] *= -1.0;
                    }
                    else { // symmetric case
                      constraints[i][j] *= 1.0;
                    }

                } // rof : jy

              start_j += (degree - 1) * (degree - 1);

              // type 2:
              for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {
                  unsigned int j = start_j + jx + ( jy * (degree - 1) );
                  if ( direction_child_intern[line_i] != mother_flip_x )
                    if ( ( jx + tmp_i ) % 2 == 0 ) { // anti-symmetric case
                      constraints[i][j] *= -1.0;
                    }
                    else { // symmetric case
                      constraints[i][j] *= 1.0;
                    }

                }
              start_j += (degree - 1) * (degree - 1);

              // type 3.1:
                // TODO: skip
                start_j += degree - 1;

              // type 3.2:
                // TODO: skip
                start_j += degree - 1;

            } // rof: block
          } // rof: DoF i

          // Face - Face
          //
          unsigned int degree_square = ( degree - 1 ) * ( degree - 1 );

          {  // Face
          unsigned int i = 12 * fe.n_dofs_per_line();
          for ( unsigned int block = 0; block < 8; block++ ) {
            unsigned int subblock      = block % 2;
            unsigned int current_block = block / 2;
            unsigned int subblock_size = fe.n_dofs_per_quad() / 2;

            // check if the counting of the DoFs is correct:
            if ( subblock == 0 && i != 12 * fe.n_dofs_per_line() + current_block * fe.n_dofs_per_quad() )
              std::cout << "ERROR: WIR HABEN UNS VERZÄHLT!!!11!EINSEINHUNDEREILF!!!1" << std::endl;

            // Type 1:
            for ( unsigned int iy = 0; iy < degree - 1; iy++)
              for ( unsigned int ix = 0; ix < degree - 1; ix++ ) {

                // Type 1 on mother:
                unsigned int j = 4 * fe.n_dofs_per_line() + subblock * subblock_size;
                for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                  for (unsigned int jx = 0; jx < degree - 1; jx++ ) {

                    if ( child_flip_x[current_block] != mother_flip_x ) //  x - direction (x-flip)
                      if ( ( ix + jx ) % 2 == 0 ) { // symmetric in x
                        constraints[i][j] *= 1.0;
                      }
                      else { // anti-symmetric in x
                        constraints[i][j] *= -1.0;
                      }

                    if ( child_flip_y[current_block] != mother_flip_y ) // y - direction (y-flip)
                      if ( ( iy + jy ) % 2 == 0 ) { // symmetric in y
                        constraints[i][j] *= 1.0;
                      }
                      else { // anti-symmetric in y
                        constraints[i][j] *= -1.0;
                      }

                    j++;
                  } // rof: Dof j
                i++;
              } // rof: DoF i


            // Type 2:
            for ( unsigned int iy = 0; iy < degree - 1; iy++)
              for ( unsigned int ix = 0; ix < degree - 1; ix++ ) {

                // Type 2 on mother:
                unsigned int j = 4 * fe.n_dofs_per_line() + degree_square + subblock * subblock_size;
                for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                  for (unsigned int jx = 0; jx < degree - 1; jx++ ) {

                    if ( child_flip_x[current_block] != mother_flip_x ) //  x - direction (x-flip)
                      if ( ( ix + jx ) % 2 == 0 ) { // symmetric in x
                        constraints[i][j] *= 1.0;
                      }
                      else { // anti-symmetric in x
                        constraints[i][j] *= -1.0;
                      }

                    if ( child_flip_y[current_block] != mother_flip_y ) // y - direction (y-flip)
                      if ( ( iy + jy ) % 2 == 0 ) { // symmetric in y
                        constraints[i][j] *= 1.0;
                      }
                      else { // anti-symmetric in y
                        constraints[i][j] *= -1.0;
                      }

                    j++;
                  } // rof: Dof j

                i++;
              } // rof: DoF i


            // Type 3 (y): 
            for ( unsigned int iy = 0; iy < degree - 1; iy++ ) {

                // Type 2 on mother:
                unsigned int j = 4 * fe.n_dofs_per_line() + degree_square + subblock * subblock_size;
                for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                  for (unsigned int jx = 0; jx < degree - 1; jx++ ) {

                    if ( child_flip_x[current_block] != mother_flip_x ) //  x - direction (x-flip)
                      if ( ( jx ) % 2 == 0 ) { // anti-symmetric in x
                        constraints[i][j] *= -1.0;
                      }
                      else { // symmetric in x
                        constraints[i][j] *= 1.0;
                      }

                    if ( child_flip_y[current_block] != mother_flip_y ) // y - direction (y-flip)
                      if ( ( iy + jy ) % 2 == 0 ) { // symmetric in y
                        constraints[i][j] *= 1.0;
                      }
                      else { // anti-symmetric in y
                        constraints[i][j] *= -1.0;
                      }

                    j++;
                  } // rof: DoF j

                // Type 3 on mother:
                j = 4 * fe.n_dofs_per_line() + 2 * degree_square + subblock * subblock_size;
                for ( unsigned int jy = 0; jy < degree - 1; jy++ ) {

                  if ( child_flip_y[current_block] != mother_flip_y ) // y - direction (y-flip)
                    if ( ( iy + jy ) % 2 == 0 ) { // symmetric in y
                      constraints[i][j] *= 1.0;
                    }
                    else { // anti-symmetric in y
                      constraints[i][j] *= -1.0;
                    }

                  j++;
                } // rof: DoF j
                i++;
            } // rof: Type 3 (y)

            // Type 3 (x): 
            for ( unsigned int ix = 0; ix < degree - 1; ix++ ) {

                // Type 2 on mother:
                unsigned int j = 4 * fe.n_dofs_per_line() + degree_square + subblock * subblock_size;
                for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                  for (unsigned int jx = 0; jx < degree - 1; jx++ ) {


                    if ( child_flip_x[current_block] != mother_flip_x ) //  x - direction (x-flip)
                      if ( ( ix + jx ) % 2 == 0 ) { // symmetric in x
                        constraints[i][j] *= 1.0;
                      }
                      else { // anti-symmetric in x
                        constraints[i][j] *= -1.0;
                      }

                    if ( child_flip_y[current_block] != mother_flip_y ) // y - direction (y-flip)
                      if ( ( jy ) % 2 == 0 ) { // anti-symmetric in y
                        constraints[i][j] *= -1.0;
                      }
                      else { // symmetric in y
                        constraints[i][j] *= 1.0;
                      }

                    j++;
                  } // rof: Dof j

                // Type 3 on mother:
                j = 4 * fe.n_dofs_per_line() + 2 * degree_square + (degree - 1) + subblock * subblock_size;
                for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {

                  if ( child_flip_x[current_block] != mother_flip_x ) //  x - direction (x-flip)
                    if ( ( ix + jx ) % 2 == 0 ) { // symmetric in x
                      constraints[i][j] *= 1.0;
                    }
                    else { // anti-symmetric in x
                      constraints[i][j] *= -1.0;
                    }

                  j++;
                }
                i++;
            } // rof: Type 3 (x)

          } // rof: block
          }

          // Next, after we have adapted the signs in the constraint matrix, based on the directions of the edges,
          // we need to modify the constraint matrix based on the orientation of the faces (i.e. if x and y direction are
          // exchanged on the face)

          // interior edges:
          for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_line(); i++ ) {
        	//unsigned int block = i / fe.n_dofs_per_line(); // TODO Remove if unnessary???

        	// check if on the mother face x and y are permuted
        	if ( mother_flip_xy ) {

        	  // copy the constrains:
        	  std::vector<double> constraints_old(dofs_on_mother.size(), 0);
        	  for ( unsigned int j = 0; j < dofs_on_mother.size(); j++) {
        	    constraints_old[j] = constraints[i][j];
        	  }

        	  unsigned int j_start = 4 * fe.n_dofs_per_line();
        	  for ( unsigned b = 0; b < 2; b++ ) {
        	    // type 1
        	    for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                  for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {
                    unsigned int j_old = j_start + jx + ( jy * (degree - 1) );
                    unsigned int j_new = j_start + jy + ( jx * (degree - 1) );
                    constraints[i][j_new] = constraints_old[j_old];
                  }
        	    j_start += degree_square;

        	    // type 2
        	    for ( unsigned int jy = 0; jy < degree - 1; jy++ )
                  for ( unsigned int jx = 0; jx < degree - 1; jx++ ) {
                    unsigned int j_old = j_start + jx + ( jy * (degree - 1) );
                    unsigned int j_new = j_start + jy + ( jx * (degree - 1) );
                    constraints[i][j_new] = - constraints_old[j_old];
                  }
        	    j_start += degree_square;

        	    // type 3
        	    for ( unsigned int j = j_start; j < j_start + (degree - 1); j++) {
        	  	  constraints[i][j] = constraints_old[j + (degree - 1)];
        	  	  constraints[i][j + (degree - 1)] = constraints_old[j];
        	    }
        	    j_start += 2 * (degree - 1);
        	  }

        	}

          }

          {
          // faces:
          const unsigned int deg = degree - 1;

          // copy the constraints
          std::vector< std::vector<double> > constraints_old(4 * fe.n_dofs_per_quad(), std::vector<double>(fe.n_dofs_per_quad(), 0));
          for (unsigned int i = 0; i < 4 * fe.n_dofs_per_quad(); i++)
        	for ( unsigned int j = 0; j < fe.n_dofs_per_quad(); j++)
        	  constraints_old[i][j] = constraints[i + (12 * fe.n_dofs_per_line())][j + (4 * fe.n_dofs_per_line())];

          // permute rows (on child)
          for ( unsigned int child = 0; child < 4; child++ ) {
            if ( !child_flip_xy[child] )
              continue;

            unsigned int i_start_new = 12 * fe.n_dofs_per_line() + (child * fe.n_dofs_per_quad());
            unsigned int i_start_old = child * fe.n_dofs_per_quad();

            unsigned int j_start     = 4 * fe.n_dofs_per_line();

            for (unsigned int block = 0; block < 2; block++) {
              // Type 1:
              for ( unsigned int ix = 0; ix < deg; ix++ ) {
                for ( unsigned int iy = 0; iy < deg; iy++ ) {

                  for ( unsigned int j = 0; j < fe.n_dofs_per_quad(); j++ )
                    constraints[i_start_new + iy + (ix * deg)][j + j_start] = constraints_old[i_start_old + ix + (iy * deg)][j];

                } // rof: iy
              } // rof: ix
              i_start_new += deg * deg;
              i_start_old += deg * deg;

              // Type 2:
              for ( unsigned int ix = 0; ix < deg; ix++ ) {
                for ( unsigned int iy = 0; iy < deg; iy++ ) {

                  for ( unsigned int j = 0; j < fe.n_dofs_per_quad(); j++ )
                    constraints[i_start_new + iy + (ix * deg)][j + j_start] = - constraints_old[i_start_old + ix + (iy * deg)][j];

                } // rof: iy
              } // rof: ix
              i_start_new += deg * deg;
              i_start_old += deg * deg;

              // Type 3:
              for ( unsigned int ix = 0; ix < deg; ix++ ) {
                for ( unsigned int j = 0; j < fe.n_dofs_per_quad(); j++ )
                  constraints[i_start_new + ix][j + j_start] = constraints_old[i_start_old + ix + deg][j];
                for ( unsigned int j = 0; j < fe.n_dofs_per_quad(); j++ )
                  constraints[i_start_new + ix + deg][j + j_start] = constraints_old[i_start_old + ix][j];
              } // rof: ix

              i_start_new += 2 * deg;
              i_start_old += 2 * deg;

            } // rof: block

          } // rof: child

          // update the constraints_old
          for (unsigned int i = 0; i < 4 * fe.n_dofs_per_quad(); i++)
        	for ( unsigned int j = 0; j < fe.n_dofs_per_quad(); j++)
        	  constraints_old[i][j] = constraints[i + (12 * fe.n_dofs_per_line())][j + (4 * fe.n_dofs_per_line())];

          // Mother
          if ( mother_flip_xy ){

        	unsigned int i_start     = 12 * fe.n_dofs_per_line();

            unsigned int j_start_new = 4 * fe.n_dofs_per_line();
            unsigned int j_start_old = 0;

            for ( unsigned int block = 0; block < 2; block++ ) {

              // Type 1:
              for ( unsigned int jx = 0; jx < deg; jx++ ) {
                for ( unsigned int jy = 0; jy < deg; jy++ ) {

                  for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_quad(); i++ )
                    constraints[i + i_start][j_start_new + jy + (jx * deg)] = constraints_old[i][j_start_old + jx + (jy * deg)];

                } // rof: jy
              } // rof: jx
              j_start_new += deg * deg;
              j_start_old += deg * deg;

              // Type 2:
              for ( unsigned int jx = 0; jx < deg; jx++ ) {
                for ( unsigned int jy = 0; jy < deg; jy++ ) {

                  for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_quad(); i++ )
                    constraints[i + i_start][j_start_new + jy + (jx * deg)] = - constraints_old[i][j_start_old + jx + (jy * deg)];

                } // rof: jy
              } // rof: jx
              j_start_new += deg * deg;
              j_start_old += deg * deg;

              // Type 3:
              for ( unsigned int jx = 0; jx < deg; jx++ ) {
                for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_quad(); i++ ) {
                  constraints[i + i_start][j_start_new + jx] = constraints_old[i][j_start_old + jx + deg];
                  constraints[i + i_start][j_start_new + jx + deg] = constraints_old[i][j_start_old + jx];
                }
              } // rof: jx
              j_start_new += 2 * deg;
              j_start_old += 2 * deg;

            } // rof: block
          } // end: Mother

          } // end: face

          if ( !printed ) {
            std::cout << "Number of DoFs on vertices: " << fe.n_dofs_per_vertex() << std::endl;
            std::cout << "Number of DoFs on lines   : " << fe.n_dofs_per_line() << std::endl;
            std::cout << "Number of DoFs on faces   : " << fe.n_dofs_per_quad(face) << std::endl;
            printed = true;

            // direction mother
            std::cout << std::endl;
            std::cout << "direction exterior mother:" << std::endl;
            for ( unsigned int i = 0; i < 4; i++ )
            	std::cout << direction_mother[i] << " ";
            std::cout << std::endl;

            // direction child:
            std::cout << std::endl;
            std::cout << "direction exterior children:" << std::endl;
            for ( unsigned int i = 0; i < 8; i++ )
            	std::cout << direction_child[i] << " ";
            std::cout << std::endl;
            std::cout << "direction interior children:" << std::endl;
            for ( unsigned int i = 0; i < 4; i++ )
            	std::cout << direction_child_intern[i] << " ";
            std::cout << std::endl;
            std::cout << std::endl;

             //default
             unsigned int min_row = 0;
             unsigned int max_row = 12 * fe.n_dofs_per_line() + 4 * fe.n_dofs_per_quad(face);

             // only one block
             //unsigned int min_row = 12 * fe.n_dofs_per_line();
             //unsigned int max_row = 12 * fe.n_dofs_per_line() + (fe.n_dofs_per_quad(face) / 2);

             // default
             unsigned int min_i = 0;
             unsigned int max_i = dofs_on_mother.size();

             // only one block
             //unsigned int min_i = 4 * fe.n_dofs_per_line();
             //unsigned int max_i = 4  * fe.n_dofs_per_line() + (fe.n_dofs_per_quad(face) / 2);

              std::cout << std::endl << "weights: " << std::endl;
              for ( unsigned int row = min_row; row < max_row; row++ ) {
                for ( unsigned int i = min_i; i < max_i; i++)
                  if ( std::abs( fe.constraints()(row, i) - constraints[row][i] ) < 0.001 )
                	if ( std::abs(constraints[row][i]) < 0.001 )
                		std::cout << 0  << "\t";
                	else
                		std::cout << constraints[row][i]  << "\t";
                  else
                	if ( std::abs(constraints[row][i]) < 0.001 )
                		std::cout << "\033[31m" << 0 << "\033[0m" << "\t";
                	else
                		std::cout << "\033[31m" << constraints[row][i] << "\033[0m" << "\t";
                std::cout << std::endl;
              }

           }

          // Apply the constraints
          // Interior edges:
          for ( unsigned int line = 0; line < 4; ++line ) {

            unsigned int row_start = line * fe.n_dofs_per_line();

            for ( unsigned int row = 0; row < fe.n_dofs_per_line(); ++row ) {
              hanging_node_constraints.add_line(dofs_on_children[row_start + row]);
              for ( unsigned int i = 0; i < dofs_on_mother.size(); ++i ) {
                hanging_node_constraints.add_entry ( 
                  dofs_on_children[row_start + row], 
                  dofs_on_mother[i], 
                  constraints[row_start + row][i]
                );
              }
              hanging_node_constraints.set_inhomogeneity(dofs_on_children[row_start + row], 0.);
            }
          } // rof: lines

          // Exterior edges
          for ( unsigned int line = 0; line < 8; ++line ) {

            unsigned int row_start = ( 4 * fe.n_dofs_per_line() ) + ( line * fe.n_dofs_per_line() );

            for ( unsigned int row = 0; row < fe.n_dofs_per_line(); ++row ) {
              hanging_node_constraints.add_line(dofs_on_children[row_start + row]);
              for ( unsigned int i = 0; i < dofs_on_mother.size(); ++i ) {
                hanging_node_constraints.add_entry( 
                  dofs_on_children[row_start + row], 
                  dofs_on_mother[i], 
                  constraints[row_start + row][i]
                );
              }
              hanging_node_constraints.set_inhomogeneity(dofs_on_children[row_start + row], 0.);
            }

          } // rof: lines

          // Faces:
          for ( unsigned int f = 0; f < 4; ++f ) {

            unsigned int row_start = ( 12 * fe.n_dofs_per_line() ) + ( f * fe.n_dofs_per_quad() );

            for ( unsigned int row = 0; row < fe.n_dofs_per_quad(); ++row ) {
              hanging_node_constraints.add_line(dofs_on_children[row_start + row]);

                for ( unsigned int i = 0; i < dofs_on_mother.size(); ++i ) {
                  hanging_node_constraints.add_entry( 
                    dofs_on_children[row_start + row], 
                    dofs_on_mother[i], 
                    constraints[row_start + row][i]
                  );
                }

              hanging_node_constraints.set_inhomogeneity(dofs_on_children[row_start + row], 0.);
            }

          } // rof: Faces

    } // rof: faces

  } // rof: cells
}

template <int dim>
void ultra_bullshit_hanging_nodes_constraints_3d(
  const DoFHandler<dim> &dof_handler,
  AffineConstraints<double> &hanging_node_constraints
) {

  std::vector<types::global_dof_index> dofs_on_mother;
  std::vector<types::global_dof_index> dofs_on_children;

  std::cout << std::endl;
  bool print = true;

  // loop over all cells
  for ( const auto &cell : dof_handler.active_cell_iterators() ) {

    // Skip artifical cells:
      // artificial cells can at best neighbor ghost cells, but we're not
      // interested in these interfaces
      if ( cell->is_artificial() )
        continue;

    // Loop over all faces
    for ( const unsigned int face : cell->face_indices() ) {

      // Skip cells without children
      if ( cell->face(face)->has_children() == false )
        continue;

      // Skip the checks!
      // TODO: Add the checks!

      // Get the FE
      const FiniteElement<dim> &fe       = cell->get_fe();
      const unsigned int        fe_index = cell->active_fe_index();

      // Get the number of DoFs on mother and children
        // Number of DoFs on the mother
        const unsigned int n_dofs_on_mother = fe.n_dofs_per_face(face);
        dofs_on_mother.resize(n_dofs_on_mother);

        // Number of DoFs on the children
        const unsigned int n_dofs_on_children = ( 12 * fe.n_dofs_per_line() + 4 * fe.n_dofs_per_quad(face) );
        // we might not use all of those in case of artificial cells, so
        // do not resize(), but reserve() and use push_back later.
        dofs_on_children.clear();
        dofs_on_children.reserve(n_dofs_on_children);

      // Skip Assert: Check that the number of constraonts match
      // TODO: Add the assert!

      // Get the current face
      const typename DoFHandler<3, 3>::face_iterator this_face = cell->face(face);

      // Fill DoFs:
      // Fill the DoFs on the mother:
        unsigned int next_index = 0;

        // DoFs on vertices
        //   The nedelec elements have no DoFs on the vertices

        // DoFs on lines
        for ( unsigned int line = 0; line < 4; ++line )
          for ( unsigned int dof = 0; dof != fe.n_dofs_per_line(); ++dof )
            dofs_on_mother[next_index++] = this_face->line(line)->dof_index(dof, fe_index);

        // DoFs on the face
        for ( unsigned int dof = 0; dof != fe.n_dofs_per_quad(face); ++dof )
          dofs_on_mother[next_index++] = this_face->dof_index(dof, fe_index);

        // Check that we have addded all DoFs
        AssertDimension(next_index, dofs_on_mother.size());

      // Fill the DoF on the children::

        // DoFs on vertices
        //   The nedelec elements have no DoFs on the vertices

        // DoFs on lines
          // next the dofs on the lines interior to the face; the order of
          // these lines is laid down in the FiniteElement class
          // documentation
          for ( unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof )
            dofs_on_children.push_back( this_face->child(0)->line(1)->dof_index(dof, fe_index) );

          for (unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof)
            dofs_on_children.push_back( this_face->child(2)->line(1)->dof_index(dof, fe_index) );

          for (unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof)
            dofs_on_children.push_back( this_face->child(0)->line(3)->dof_index(dof, fe_index) );

          for (unsigned int dof = 0; dof < fe.n_dofs_per_line(); ++dof)
            dofs_on_children.push_back( this_face->child(1)->line(3)->dof_index(dof, fe_index) );

          // dofs on the bordering lines
          for ( unsigned int line = 0; line < 4; ++line )
            for ( unsigned int child = 0; child < 2; ++child )
              for (unsigned int dof = 0; dof != fe.n_dofs_per_line(); ++dof )
                dofs_on_children.push_back( this_face->line(line)->child(child)->dof_index(dof, fe_index) );

        // DoFs on faces
          for ( unsigned int child = 0; child < 4; ++child ) {

            // skip artificial cells
            if ( cell->neighbor_child_on_subface(face, child)->is_artificial() )
              continue;

            for ( unsigned int dof = 0; dof != fe.n_dofs_per_quad(face); ++dof )
              dofs_on_children.push_back( this_face->child(child)->dof_index(dof, fe_index) );
          }

        // Check that we have added all DoFs
          // note: can get fewer DoFs when we have artificial cells:
          Assert(dofs_on_children.size() <= n_dofs_on_children,
          ExcInternalError());

		// The constraint matrix
          std::vector<std::vector<double>>
            constraints(12 * fe.n_dofs_per_line() + 4 * fe.n_dofs_per_quad(), std::vector<double>(dofs_on_mother.size(), 0));

        { // copy the constraint matrix
            // copy the wights for the interior edges
            for ( unsigned int line = 0; line < 4; line++ ) {
              unsigned int row_start  = line * fe.n_dofs_per_line();
              for ( unsigned int row = 0; row < fe.n_dofs_per_line(); row++ ) {
                for ( unsigned int i = 0; i < 4 * fe.n_dofs_per_line(); i++ )
                  constraints[row + row_start][i] = fe.constraints()(row + row_start, i);
//                  constraints[row + row_start][i] = 0;
              }
            }

            // copy the wights for the interior edges
            for ( unsigned int line = 0; line < 4; line++ ) {
              unsigned int row_start  = line * fe.n_dofs_per_line();
              for ( unsigned int row = 0; row < fe.n_dofs_per_line(); row++ ) {
                for ( unsigned int i = 4 * fe.n_dofs_per_line(); i < dofs_on_mother.size(); i++ )
                  constraints[row + row_start][i] = fe.constraints()(row + row_start, i);
//                  constraints[row + row_start][i] = 0;
              }
            }

            // copy the weights for exterior edges
            unsigned int row_offset = 4 * fe.n_dofs_per_line();
            for ( unsigned int line = 0; line < 8; line++ ) {
              unsigned int row_start = line * fe.n_dofs_per_line();
              for ( unsigned int row = row_offset; row < row_offset + fe.n_dofs_per_line(); row++ )
                for ( unsigned int i = 0; i < dofs_on_mother.size(); i++ )
                  constraints[row + row_start][i] = fe.constraints()(row + row_start, i);
//                  constraints[row + row_start][i] = 0;
            }

            // copy the weights for the faces
            row_offset = 12 * fe.n_dofs_per_line();
            for ( unsigned int face = 0; face < 4; face++ ) {
              unsigned int row_start = face * fe.n_dofs_per_quad();
              for ( unsigned int row = row_offset; row < row_offset + fe.n_dofs_per_quad(); row++ )
                for ( unsigned int i = 0; i < dofs_on_mother.size(); i++ )
                  constraints[row + row_start][i] = fe.constraints()(row + row_start, i);
//                  constraints[row + row_start][i] = 0;
            }
          }

        if ( print ) {
            unsigned int row_offset = 12 * fe.n_dofs_per_line();
            for ( unsigned int face = 0; face < 4; face++ ) {
              unsigned int row_start = face * fe.n_dofs_per_quad();
              for ( unsigned int row = row_offset; row < row_offset + fe.n_dofs_per_quad(); row++ ) {
                for ( unsigned int i = 0; i < dofs_on_mother.size(); i++ )
                  std::cout << constraints[row + row_start][i]  << "\t";
                std::cout << std::endl;
              }
            }
          print = false;
        }


        // Apply the constraints
          // Interior edges:
          for ( unsigned int line = 0; line < 4; ++line ) {

            unsigned int row_start = line * fe.n_dofs_per_line();

            for ( unsigned int row = 0; row < fe.n_dofs_per_line(); ++row ) {
              hanging_node_constraints.add_line(dofs_on_children[row_start + row]);
              for ( unsigned int i = 0; i < dofs_on_mother.size(); ++i ) {
                hanging_node_constraints.add_entry (
                  dofs_on_children[row_start + row],
                  dofs_on_mother[i],
                  constraints[row_start + row][i]
                );
              }
              hanging_node_constraints.set_inhomogeneity(dofs_on_children[row_start + row], 0.);
            }
          } // rof: lines

          // Exterior edges
          for ( unsigned int line = 0; line < 8; ++line ) {

            unsigned int row_start = ( 4 * fe.n_dofs_per_line() ) + ( line * fe.n_dofs_per_line() );

            for ( unsigned int row = 0; row < fe.n_dofs_per_line(); ++row ) {
              hanging_node_constraints.add_line(dofs_on_children[row_start + row]);
              for ( unsigned int i = 0; i < dofs_on_mother.size(); ++i ) {
                hanging_node_constraints.add_entry(
                  dofs_on_children[row_start + row],
                  dofs_on_mother[i],
                  constraints[row_start + row][i]
                );
              }
              hanging_node_constraints.set_inhomogeneity(dofs_on_children[row_start + row], 0.);
            }

          } // rof: lines

          // Faces:
          for ( unsigned int f = 0; f < 4; ++f ) {

            unsigned int row_start = ( 12 * fe.n_dofs_per_line() ) + ( f * fe.n_dofs_per_quad() );

            for ( unsigned int row = 0; row < fe.n_dofs_per_quad(); ++row ) {
              hanging_node_constraints.add_line(dofs_on_children[row_start + row]);

                for ( unsigned int i = 0; i < dofs_on_mother.size(); ++i ) {
                  hanging_node_constraints.add_entry(
                    dofs_on_children[row_start + row],
                    dofs_on_mother[i],
                    constraints[row_start + row][i]
                  );
                }

              hanging_node_constraints.set_inhomogeneity(dofs_on_children[row_start + row], 0.);
            }

          } // rof: Faces

    } // rof: faces

  } // rof: cells

}

} // namespace KirasFM
#endif
