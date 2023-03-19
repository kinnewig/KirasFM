//
// Created by sebastian on 25.02.23.
//

#ifndef KIRASFM_REFINEMENT_COMMUNICATOR_H
#define KIRASFM_REFINEMENT_COMMUNICATOR_H

#include <deal.II/base/tensor.h>
#include <iostream>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

namespace KirasFM {
  using namespace dealii;

template <int dim>
class RefinementCommunicator {
  public:
    // Constructor
    RefinementCommunicator ();
    RefinementCommunicator ( unsigned int n_domains_ );
    RefinementCommunicator ( const RefinementCommunicator<dim>& copy );

    // return functions
    std::vector<bool>
    coarsen ( unsigned int i, unsigned int j);

    std::vector<bool>
    refinement ( unsigned int i, unsigned int j);

    // update functions
    void coarsen (
      std::vector<bool> in,
      unsigned int i,
      unsigned int j
    );

    void refinement (
      std::vector<bool> in,
      unsigned int i,
      unsigned int j
    );

    void update ( RefinementCommunicator<dim> rc, unsigned int i );

    // copy assignment operator
    RefinementCommunicator<dim>& operator= ( const RefinementCommunicator<dim>& copy );

  private:
    // Befriend the class with boost::serialization for the use with MPI
    friend class boost::serialization::access;

    // Function needed by boost::serialization
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar& n_domains;
      ar& coarsen_data;
      ar& refinement_data;

      v = version;
    }
    unsigned int n_domains;
    unsigned int v;

    std::vector<
      std::vector<bool>
    > coarsen_data;

    std::vector<
      std::vector<bool>
    > refinement_data;
};

} //namespace KirasFM

#endif //KIRASFM_REFINEMENT_COMMUNICATOR_H
