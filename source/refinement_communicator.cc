//
// Created by sebastian on 25.02.23.
//

#include <refinement_communicator.h>

namespace KirasFM{
   using namespace dealii;

   // Default Constructor
   template<int dim>
   RefinementCommunicator<dim>::RefinementCommunicator () {

   }

   // Constructor
   template<int dim>
   RefinementCommunicator<dim>::RefinementCommunicator ( unsigned int n_domains_ ) {
     n_domains = n_domains_;

     // initialize values
     std::vector<bool> tmp_vec;

     for ( unsigned int i = 0; i < n_domains * n_domains; i++ ) {
       coarsen_data.push_back(tmp_vec);
       refinement_data.push_back(tmp_vec);
     }
   }

   // Copy Constructor (Rule of 3: 2/3)
   template<int dim>
   RefinementCommunicator<dim>::RefinementCommunicator ( const RefinementCommunicator<dim> &copy ) {
     n_domains       = copy.n_domains;

     coarsen_data    = copy.refinement_data;
     refinement_data = copy.refinement_data;
   }

   // return functions
   template <int dim>
   std::vector<bool>
   RefinementCommunicator<dim>::coarsen ( unsigned int i, unsigned int j ) {
     AssertIndexRange( i, n_domains );
     AssertIndexRange( j, n_domains );
     return refinement_data[ (i * n_domains) + j ];
   }

   template <int dim>
   std::vector<bool>
   RefinementCommunicator<dim>::refinement ( unsigned int i, unsigned int j ) {
     AssertIndexRange( i, n_domains );
     AssertIndexRange( j, n_domains );
     return refinement_data[ (i * n_domains) + j ];
   }

   // update functions
   template<int dim>
   void RefinementCommunicator<dim>::coarsen (
     std::vector<bool> in,
     unsigned int i,
     unsigned int j
   ) {
     AssertIndexRange( i, n_domains );
     AssertIndexRange( j, n_domains );
     coarsen_data[(i * n_domains) + j] = in;
   }

   template<int dim>
   void RefinementCommunicator<dim>::refinement (
     std::vector<bool> in,
     unsigned int i,
     unsigned int j
   ) {
     AssertIndexRange( i, n_domains );
     AssertIndexRange( j, n_domains );
     refinement_data[(i * n_domains) + j] = in;
   }

    template<int dim>
    void RefinementCommunicator<dim>::update ( RefinementCommunicator<dim> rc, unsigned int i ) {
      AssertIndexRange( i, n_domains );
      for ( unsigned int j = 0; j < n_domains; j++ ) {
        coarsen_data[ (i * n_domains) + j ] = rc.coarsen(i, j);
        refinement_data[ (i * n_domains) + j ] = rc.refinement(i, j);
      }
    }

    // Assignment Operator (Rule of 3: 3/3)
    template<int dim>
    RefinementCommunicator<dim>& RefinementCommunicator<dim>::operator= (
      const RefinementCommunicator<dim> &copy
    ) {
      if (this == &copy)
        return *this;

      n_domains  = copy.n_domains;

      coarsen_data    = copy.coarsen_data;
      refinement_data = copy.refinement_data;

      return *this;
    }

    template class RefinementCommunicator<3>;
}