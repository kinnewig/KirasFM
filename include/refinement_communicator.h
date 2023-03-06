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
    RefinementCommunicator();
    RefinementCommunicator(unsigned int n_domains_);
    RefinementCommunicator(const RefinementCommunicator<dim>& copy);

    // functions
    std::vector<std::vector<bool>> refinement();
    std::vector<bool> refinement(unsigned int i);
    void update(std::vector<bool> data, unsigned int i);
    void update(RefinementCommunicator<dim> rc, unsigned int i);

    // Operator
    RefinementCommunicator<dim>& operator=(const RefinementCommunicator<dim>& copy);

  private:
    // Befriend the class with boost::serialization for the use with MPI
    friend class boost::serialization::access;

    // Function needed by boost::serialization
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar& n_domains;
      ar& refinement_data;
      v = version;
    }
    unsigned int n_domains;
    unsigned int v;

    std::vector<std::vector<bool>> refinement_data;
};

} //namespace KirasFM

#endif //KIRASFM_REFINEMENT_COMMUNICATOR_H
