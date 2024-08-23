#include <boundary_values.h>

namespace KirasFM {
  //const double PI  = 3.141592653589793;
  //const double PI2 = 9.869604401089359;

  // here we are using quite artificial boundary values,
  // the benefit is, that we know the exact solution for the
  // electric field E, i.e. this is also the exact solution 
  // to the partial differential equation we aim to solve
  template<>
  void DirichletBoundaryValues<2>::vector_value (
    const Point<2> &p,
    Vector<double> &values
  ) const {
    double size = 0.01;

    values(0) = std::exp( - ( std::pow(p(0) - 0.5, 2) / size) );
    values(1) = 0.0;
    values(2) = 0.0; 
    values(3) = 0.0;
  }

  template<>
  void DirichletBoundaryValues<3>::vector_value (
    const Point<3> &p,
    Vector<double> &values
  ) const {
    double size = 0.01;

    values(0) = std::exp( - ( std::pow(p(0) - 0.5, 2) / size)  - (std::pow(p(2) - 0.5, 2) / size) );
    values(1) = 0.0;
    values(2) = 0.0; 
    values(3) = 0.0;
    values(4) = 0.0;
    values(5) = 0.0;
  }

  template<>
  void CurlRHS<2>::vector_value (
    const Point<2> &/*p*/,
    Vector<double> &values
  ) const {
    values(0) = 0.0;
    values(1) = 0.0; 
    values(2) = 0.0;
    values(3) = 0.0;
  }

  template<>
  void CurlRHS<3>::vector_value (
    const Point<3> &/*p*/,
    Vector<double> &values
  ) const {
    values(0) = 0.0;
    values(1) = 0.0; //- PI2 * sin( PI * p(0) ) * cos( PI * p(1) );
    values(2) = 0.0;
    values(3) = 0.0;
    values(4) = 0.0;
    values(5) = 0.0;
  }

} // namespace KirasFM
