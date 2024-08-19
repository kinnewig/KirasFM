#include <surface_communicator.h>

namespace KirasFM{
  using namespace dealii;

  template <int dim>
  SurfaceCommunicator<dim>::SurfaceCommunicator() {

  }

  template <int dim>
  SurfaceCommunicator<dim>::SurfaceCommunicator ( unsigned int n_domains_ ) {
    n_domains = n_domains_;

    // initialize value
    std::vector<
      std::vector<Tensor<1, dim, std::complex<double>>>
    > tmp_value;

    std::vector<
      std::vector<std::complex<double>>
    > tmp_curl;

    for ( unsigned int i = 0; i < n_domains * n_domains; i++ ) {
      value_data.push_back(tmp_value);
      curl_data.push_back(tmp_curl);
    }

  }

  // Copy Constructor (Rule of 3: 2/3)
  template <int dim>
  SurfaceCommunicator<dim>::SurfaceCommunicator ( const SurfaceCommunicator<dim>& copy ) {
    n_domains  = copy.n_domains;

    value_data = copy.value_data;
    curl_data   = copy.curl_data;
  }

  // return functions
  template <int dim>
  std::vector<
    std::vector<Tensor<1, dim, std::complex<double>>>
  > SurfaceCommunicator<dim>::value ( unsigned int i, unsigned int j ) {
    AssertIndexRange( i, n_domains );
    AssertIndexRange( j, n_domains );
    return value_data[ (i * n_domains) + j ];
  }

  template <int dim>
  std::vector<
    std::vector<std::complex<double>>
  > SurfaceCommunicator<dim>::curl(unsigned int i, unsigned int j) {
    AssertIndexRange( i, n_domains );
    AssertIndexRange( j, n_domains );
    return curl_data[( i * n_domains) + j ];
  }

  template <int dim>
  void SurfaceCommunicator<dim>::value(
    std::vector<std::vector<Tensor<1, dim, std::complex<double>>>> in,
    unsigned int i,
    unsigned int j
  ) {
    AssertIndexRange( i, n_domains );
    AssertIndexRange( j, n_domains );
    value_data[(i * n_domains) + j] = in;
  }

  template <int dim>
  void SurfaceCommunicator<dim>::curl(
    std::vector<std::vector<std::complex<double>>> in,
    unsigned int i,
    unsigned int j
  ) {
    AssertIndexRange(i, n_domains);
    AssertIndexRange(j, n_domains);
    curl_data[(i * n_domains) + j] = in;
  }


  template <int dim>
  void SurfaceCommunicator<dim>::update( SurfaceCommunicator<dim> sc, unsigned int i ) {
    AssertIndexRange( i, n_domains );
    for ( unsigned int j = 0; j < n_domains; j++ ) {
      value_data[ (i * n_domains) + j ] = sc.value(i, j);
      curl_data[ (i * n_domains) + j ]  = sc.curl(i, j);
    }
  }


  // Assaignment operator (Rule of 3: 3/3)
  template <int dim>
  SurfaceCommunicator<dim>& SurfaceCommunicator<dim>::operator=(
    const SurfaceCommunicator<dim>& copy
  ) {
    if (this == &copy)
      return *this;

    n_domains  = copy.n_domains;

    value_data = copy.value_data;
    curl_data   = copy.curl_data;

    return *this;
  }

  template class SurfaceCommunicator<2>;
} // namespace: KirasFM
