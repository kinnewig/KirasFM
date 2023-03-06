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
    // standart RHS (analythical solution is not known)
    values(0) = exp(- std::pow( p(0) - 0.5, 2 ) / 0.01 );
    values(1) = 0.0;
    values(2) = 0.0; 
    values(3) = 0.0;
  }

  template<>
  void DirichletBoundaryValues<3>::vector_value (
    const Point<3> &p,
    Vector<double> &values
  ) const {

    //values(0) = std::exp( - ( std::pow(p(0) - 0.5, 2) / 0.1)  - (std::pow(p(2) - 0.5, 2) / 0.1) );
    //values(1) = 0.0;
    //values(2) = 0.0;
    //values(3) = std::exp( - ( std::pow(p(0) - 0.5, 2) / 0.1)  - (std::pow(p(2) - 0.5, 2) / 0.1) );
    //values(4) = 0.0;
    //values(4) = 0.0;

    values(0) = ( 1.0 / std::sqrt(2) ) * std::exp( - ( std::pow(p(0) - 0.5, 2) / 0.035)  - (std::pow(p(1) - 0.5, 2) / 0.035) );
    values(1) = 0.0;
    values(2) = 0.0;
    values(3) = ( 1.0 / std::sqrt(2) ) * std::exp( - ( std::pow(p(0) - 0.5, 2) / 0.035)  - (std::pow(p(1) - 0.5, 2) / 0.035) );
    values(4) = 0.0;
    values(5) = 0.0;

//    // === Silverball === 
//    values(0) = 1.0;
//    values(1) = 1.0;
//    values(2) = 1.0;
//    values(3) = 1.0;
//    values(4) = 1.0;
//    values(5) = 1.0;

//    // === Diamond Fin ===
//    double block_size = 1.0;
//    double center_x = 3.0 * block_size + 1.0 + 0.5;
//    double center_y = (5.0 + 1.0) * block_size + 1.25;
//    double beamwidth = 0.5;
//    values(0) = std::exp( - ( std::pow(p(0) - center_x, 2) / (beamwidth * 0.5) )  - (std::pow(p(1) - center_y, 2) / beamwidth) );
//    values(1) = 0.0;
//    values(2) = 0.0;
//    values(3) = std::exp( - ( std::pow(p(0) - center_x, 2) / (beamwidth * 0.5) )  - (std::pow(p(1) - center_y, 2) / beamwidth) );
//    values(4) = 0.0;
//    values(5) = 0.0;


//  // === Gallium Laser ===
//  double block_size = 0.600;
//  double center_x   = 5.50 * block_size;
//  double center_z   = 1.05 * block_size;
//  double beamwidth  = 0.05;
//  values(0) = std::exp( - ( std::pow(p(0) - center_x, 2) / beamwidth )  - (std::pow(p(2) - center_z, 2) / beamwidth) );
//  values(1) = 0.0;
//  values(2) = std::exp( - ( std::pow(p(0) - center_x, 2) / beamwidth )  - (std::pow(p(2) - center_z, 2) / beamwidth) );
//  values(3) = std::exp( - ( std::pow(p(0) - center_x, 2) / beamwidth )  - (std::pow(p(2) - center_z, 2) / beamwidth) );
//  values(4) = 0.0;
//  values(5) = std::exp( - ( std::pow(p(0) - center_x, 2) / beamwidth )  - (std::pow(p(2) - center_z, 2) / beamwidth) );

  }

  template<>
  void CurlRHS<2>::vector_value (
    const Point<2> &p,
    Vector<double> &values
  ) const {
    values(0) = p(0) + p(1); // so the compiler shuts up
    values(0) = 0.0;
    values(1) = 0.0; 
    values(2) = 0.0;
    values(3) = 0.0;
  }

  template<>
  void CurlRHS<3>::vector_value (
    const Point<3> &p,
    Vector<double> &values
  ) const {
    values(0) = p(0) + p(1) + p(2); // so the compiler shuts up
    values(0) = 0.0;
    values(1) = 0.0; //- PI2 * sin( PI * p(0) ) * cos( PI * p(1) );
    values(2) = 0.0;
    values(3) = 0.0;
    values(4) = 0.0;
    values(5) = 0.0;
  }

} // namespace KirasFM
