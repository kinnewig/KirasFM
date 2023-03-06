#ifndef SURFACE_COMMUNICATOR_H
#define SURFACE_COMMUNICATOR_H

#include <deal.II/base/tensor.h>
#include <iostream>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

namespace KirasFM{
  using namespace dealii;

template <int dim>
class SurfaceCommunicator {
  public:
  
    SurfaceCommunicator();

    SurfaceCommunicator(unsigned int n_domains_);

    SurfaceCommunicator(const SurfaceCommunicator<dim>& copy);

    std::vector<
      std::vector<Tensor<1, dim, std::complex<double>>>
    > value(unsigned int i, unsigned int j);

    std::vector<
      std::vector<std::complex<double>>
    > curl (unsigned int i, unsigned int j);
 
    void value (
      std::vector<std::vector<Tensor<1, dim, std::complex<double>>>> in,
      unsigned int i, unsigned int j
    );

    void curl (
      std::vector<std::vector<std::complex<double>>> in,
      unsigned int i, unsigned int j
    );

    void update(SurfaceCommunicator<dim> sc, unsigned int i);

    SurfaceCommunicator<dim>& operator=(const SurfaceCommunicator<dim>& copy);


  private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar& n_domains;
        ar& n_faces;
        ar& value_data;
        ar& curl_data;
        v = version;
    }


    unsigned int n_domains;
    unsigned int n_faces;
    unsigned int v;

    std::vector<std::vector<
      std::vector<Tensor<1, dim, std::complex<double>>>
    >> value_data;

    std::vector<std::vector<
      std::vector<std::complex<double>>
    >> curl_data;
};

} // namespace KirasFM
#endif
