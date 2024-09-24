/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2024 Sebastian Kinnewig
 *
 * The code is licensed under the GNU Lesser General Public License as
 * published by the Free Software Foundation in version 2.1
 * The full text of the license can be found in the file LICENSE.md
 *
 * ---------------------------------------------------------------------
 * Contact:
 *   Sebastian Kinnewig
 *   Leibniz Universität Hannover (LUH)
 *   Institut für Angewandte Mathematik (IfAM)
 *
 * Questions?
 *   E-Mail: kinnewig@ifam.uni-hannover.de
 *
 * Date: November 2021
 *       Update: September 2024
 */

#ifndef BOUNDARY_VALUES_H
#define BOUNDARY_VALUES_H

// === deal.II includes ===
#include <deal.II/base/function.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// === C++ includes ===
#include <cmath>
#include <complex>
#include <iostream>

namespace KirasFM
{
  using namespace dealii;

  // === DirichletBoundaryValues ===
  /*
   * DirichletBoundaryValues defines the function E_{inc} in the
   * equation:
   *      trace( E ) = trace( E_{inc} ) on \Gamma_{inc}
   */
  template <int dim>
  class DirichletBoundaryValues : public Function<dim>
  {
  public:
    DirichletBoundaryValues()
      : Function<dim>(2 * dim)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); p++)
        {
          DirichletBoundaryValues<dim>::vector_value(points[p], value_list[p]);
        }
    }
  };



  // === CurlRHS ===
  /*
   * this defines the function f(x) in:
   *      curl ( curl ( E ) ) - \omega^2 E = f( x ) on \Omega
   */
  template <int dim>
  class CurlRHS : public Function<dim>
  {
  public:
    CurlRHS()
      : Function<dim>(2 * dim)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override
    {
      Assert(value_list.size() == points.size(),
             ExcDimensionMismatch(value_list.size(), points.size()));

      for (unsigned int p = 0; p < points.size(); p++)
        {
          CurlRHS<dim>::vector_value(points[p], value_list[p]);
        }
    }
  };


} // namespace KirasFM
#endif
