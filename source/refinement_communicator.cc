//
// Created by sebastian on 25.02.23.
//

#include <refinement_communicator.h>

namespace KirasFM{
    using namespace dealii;

    // Default Constructor
    template<int dim>
    RefinementCommunicator<dim>::RefinementCommunicator() {}

    // Constructor
    template<int dim>
    RefinementCommunicator<dim>::RefinementCommunicator(unsigned int n_domains_) {
      n_domains = n_domains_;

      // initialize data vector
      refinement_data = std::vector<std::vector<bool>>(n_domains_);
    }

    // Copy Constructor (Rule of 3: 2/3)
    template<int dim>
    RefinementCommunicator<dim>::RefinementCommunicator(const RefinementCommunicator<dim> &copy) {
      n_domains       = copy.n_domains;
      refinement_data = copy.refinement_data;
    }

    template<int dim>
    std::vector<std::vector<bool>> RefinementCommunicator<dim>::refinement() {
      return refinement_data;
    }

    template<int dim>
    std::vector<bool> RefinementCommunicator<dim>::refinement(unsigned int i) {
      return refinement_data[i];
    }

    template<int dim>
    void RefinementCommunicator<dim>::update(RefinementCommunicator<dim> rc, unsigned int i) {
      AssertIndexRange(i, n_domains);
      refinement_data[i] = rc.refinement(i);
    }

    template<int dim>
    void RefinementCommunicator<dim>::update(std::vector<bool> data, unsigned int i) {
      AssertIndexRange(i, n_domains);
      refinement_data[i] = data;
    }

    // Assignment Operator (Rule of 3: 3/3)
    template<int dim>
    RefinementCommunicator<dim>& RefinementCommunicator<dim>::operator=(const RefinementCommunicator<dim> &copy) {
      if ( this == &copy )
        return *this;

      n_domains       = copy.n_domains;
      refinement_data = copy.refinement_data;

      return *this;
    }

    template class RefinementCommunicator<3>;
}