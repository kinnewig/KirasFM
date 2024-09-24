/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 *  Written by Sebastian Kinnewig
 *  November 2021
 *
 */

#include <maxwell_solver.h>

namespace KirasFM
{
  using namespace dealii;

  // === trace: normal x curl u ===
  template <int dim, typename Number1, typename Number2>
  Tensor<1, dim, typename ProductType<Number1, Number2>::type>
  trace(const Tensor<1, (dim == 2) ? 1 : 3, Number1> &u,
        const Tensor<1, dim, Number2>                &n)
  {
    Tensor<1, dim, typename ProductType<Number1, Number2>::type> result;
    switch (dim)
      {
        case 2:
          result[0] = n[1] * u[0];
          result[1] = -n[0] * u[1];
          break;
        case 3:
          result[0] = n[1] * u[2] - n[2] * u[1];
          result[1] = n[2] * u[0] - n[0] * u[2];
          result[2] = n[0] * u[1] - n[1] * u[0];
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }


  // === constructor ===
  // standart constructor:
  template <int dim>
  MaxwellProblem<dim>::MaxwellProblem(ParameterReader   &param,
                                      ConditionalOStream pcout,
                                      TimerOutput        timer,
                                      const unsigned int domain_id,
                                      const unsigned int N_domains,
                                      MPI_Comm           local_mpi_comm)
    : mpi_communicator(local_mpi_comm)
    ,

    pcout(pcout)
    ,

    timer(timer)
    ,

    triangulation(
      //    mpi_communicator,
      //    typename Triangulation<dim>::MeshSmoothing (
      //      Triangulation<dim>::smoothing_on_refinement
      //      | Triangulation<dim>::smoothing_on_coarsening
      //    )
      )
    ,

    dof_handler(triangulation)
    ,

    fe(FE_NedelecSZ<dim>(
         param.get_integer("Mesh & geometry parameters", "Polynomial degree")),
       2)
    ,

    prm(param)
    ,

    // Surface Communicator
    SurfaceOperator(SurfaceCommunicator<dim>(N_domains))
    , g_out(SurfaceCommunicator<dim>(N_domains))
    ,

    RefinementOperator(RefinementCommunicator<dim>(N_domains))
    ,

    domain_id(domain_id)
    , N_domains(N_domains)
    ,

    first_rhs(true)
    , solved(false)
  {}

  // copy constructor
  template <int dim>
  MaxwellProblem<dim>::MaxwellProblem(const MaxwellProblem<dim> &copy)
    : mpi_communicator(copy.mpi_communicator)
    ,

    pcout(copy.pcout)
    ,

    timer(copy.timer)
    ,

    triangulation(
      //    mpi_communicator,
      //    typename Triangulation<dim>::MeshSmoothing (
      //      Triangulation<dim>::smoothing_on_refinement |
      //      Triangulation<dim>::smoothing_on_coarsening
      //    )
      )
    ,

    dof_handler(triangulation)
    ,

    fe(FE_NedelecSZ<dim>(copy.prm.get_integer("Mesh & geometry parameters",
                                              "Polynomial degree")),
       2)
    ,

    prm(copy.prm)
    ,

    // Surface Communicator
    SurfaceOperator(SurfaceCommunicator<dim>(copy.N_domains))
    , g_out(SurfaceCommunicator<dim>(copy.N_domains))
    ,

    RefinementOperator(RefinementCommunicator<dim>(copy.N_domains))
    ,

    domain_id(copy.domain_id)
    , N_domains(copy.N_domains)
    ,

    first_rhs(copy.first_rhs)
    , solved(copy.solved)
  {}


  template <int dim>
  void
  MaxwellProblem<dim>::setup_system()
  {
    timer.enter_subsection("Setup dof system");
    pcout << "setup system...";

    // initialize the dof handler
    dof_handler.distribute_dofs(fe);

    // Number of degrees of freedom on the current subdomain
    std::cout << "Number of degrees of freedom on domain " << domain_id << ": "
              << dof_handler.n_dofs() << std::endl;

    // get the locally owned dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Initialize the vectors:
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    rhs_backup.reinit(locally_owned_dofs, mpi_communicator);

    // Constraits
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      0 /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      dim /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    constraints.close();

    // create sparsity pattern:
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    // create the system matrix
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);

    pcout << " done!" << std::endl;
    timer.leave_subsection();
  }

  template <int dim>
  void
  MaxwellProblem<dim>::setup_system(DoFHandler<dim> &dof_handler_local)
  {
    timer.enter_subsection("Setup dof system");

    // initialize the dof handler
    dof_handler_local.distribute_dofs(fe);

    // Constraits
    // locally_owned_dofs = dof_handler_local.locally_owned_dofs();
    // DoFTools::extract_locally_relevant_dofs(dof_handler_local,
    // locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler_local, constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler_local,
      0 /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler_local,
      dim /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    constraints.close();

    timer.leave_subsection();
  }


  /*
   * Assemble the system matrix: M
   * We decompose the system into real and imaginary part.
   *
   *         |  A   B |   | Re(E) |   | Re(f) |
   * M * E = |        | . |       | = |       |
   *         | -B   A |   | Im(E) |   | Im(f) |
   *
   * where E = Re(E) + i * Im(E), where i is the imaginary unit.
   *
   * And A = \int_{cell} curl( \phi_i(x) ) * curl( \mu * phi_j(x) ) dx
   *       - \omega^2 \int_{cell} \phi_i(x) * \phi_j(x) dx
   *
   *     B = \int_{face} \Trace( \psi_i(s) ) * \Trace( \psi_j(s) ) ds
   *
   * which corresponds to the maxwell equations, with robin boundary conditions
   * in the weak form
   */
  template <int dim>
  void
  MaxwellProblem<dim>::assemble_system()
  {
    timer.enter_subsection("Assemble system matrix");
    pcout << "assemble system matrix...\t";

    if (rebuild)
      rhs_backup = system_rhs;

    system_matrix = 0;
    system_rhs    = 0;

    const unsigned int curl_dim = (dim == 2) ? 1 : 3;

    // choose the quadrature formulas
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_q_points      = quadrature_formula.size(),
                       n_face_q_points = face_quadrature_formula.size(),
                       dofs_per_cell   = fe.dofs_per_cell;

    // set update flags
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_gradients | update_hessians |
                                       update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    // create the local left hand side and right hand side
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // material constants:

    // const std::complex<double> alpha( 0.0, 1.0 / (4.0 * omega) );
    // const std::complex<double> beta( 1.0 / (3.0 * omega * omega), 0.0 );
    const std::complex<double> alpha(1.0, 0);

    // loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        // initialize values:
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        // cell dependent material constants:
        const double               mu_term = 1.0;
        const std::complex<double> omega =
          prm.get_wavenumber(cell->material_id());

        // === Domain ===
        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int block_index_i =
              fe.system_to_block_index(i).first;

            // we only want to compute this once
            std::vector<Tensor<1, dim>>      phi_i(n_q_points);
            std::vector<Tensor<1, curl_dim>> curl_phi_i(n_q_points);
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                phi_i[q_point] =
                  fe_values[vec[block_index_i]].value(i, q_point);
                curl_phi_i[q_point] =
                  fe_values[vec[block_index_i]].curl(i, q_point);
              }

            for (unsigned int j = i; j < dofs_per_cell; j++)
              { // we are using the symmetry of the here
                const unsigned int block_index_j =
                  fe.system_to_block_index(j).first;

                /*
                 * Assamble the block A, from the system matrix M
                 *
                 *  | A   0 |
                 *  |       |
                 *  | 0   A |
                 *
                 * where A = \int_{cell} curl( \phi_i(x) ) * curl( \mu *
                 * phi_j(x) ) dx
                 *         - \omega^2 \int_{cell} \phi_i(x) * \phi_j(x) dx
                 */

                double mass_part = 0;
                double curl_part = 0;

                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    Tensor<1, dim> phi_j =
                      fe_values[vec[block_index_i]].value(j, q_point);
                    Tensor<1, curl_dim> curl_phi_j =
                      fe_values[vec[block_index_i]].curl(j, q_point);

                    curl_part +=
                      curl_phi_i[q_point] * curl_phi_j * fe_values.JxW(q_point);

                    mass_part +=
                      phi_i[q_point] * phi_j * fe_values.JxW(q_point);

                  } // rof: q_point

                // Use skew-symmetry to fill matrices:
                std::complex<double> massterm =
                  (mu_term * curl_part) - ((omega * omega) * mass_part);

                if (block_index_i != block_index_j)
                  {
                    cell_matrix(i, j) = massterm.imag();
                    cell_matrix(j, i) = massterm.imag();
                  }
                else
                  {
                    cell_matrix(i, j) = massterm.real();
                    cell_matrix(j, i) = massterm.real();
                  }

              } // rof: dof_j

          } // rof: dof_i

        // === boundary condition ===
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            fe_face_values.reinit(cell, face);

            if (cell->face(face)->at_boundary() == false)
              continue;

            // --- Robin boundary conditions ---
            // We apply the robin boundary to the robin_boundary (boundary_id =
            // 0) additionally we also apply the robin boundary to the
            // dirichlet_boundary (boundary_id = 1) to avoid reflections on that
            // surface
            if (cell->face(face)->boundary_id() == 0 ||
                cell->face(face)->boundary_id() == 1)
              {
                /*
                 * Assamble the block B
                 *
                 *  |  0   B |
                 *  |        |
                 *  | -B   0 |
                 *
                 *  where B = \int_{face} \Trace( \psi_i(s) ) * \Trace(
                 * \psi_j(s) ) ds
                 */

                // compute the normal
                std::vector<Tensor<1, dim>> normal(n_face_q_points);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  normal[q_point] = fe_face_values.normal_vector(q_point);

                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;

                    if (fe.has_support_on_face(i, face) == false)
                      continue;

                    // we only want to compute this once
                    std::vector<Tensor<1, dim>> phi_i(n_face_q_points);
                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        phi_i[q_point] = CrossProduct::trace_tangential(
                          fe_face_values[vec[block_index_i]].value(i, q_point),
                          normal[q_point]);
                      }

                    for (const unsigned int j : fe_face_values.dof_indices())
                      {
                        const unsigned int block_index_j =
                          fe.system_to_block_index(j).first;

                        if (fe.has_support_on_face(j, face) == false)
                          continue;

                        double robin = 0;


                        for (unsigned int q_point = 0;
                             q_point < n_face_q_points;
                             q_point++)
                          {
                            Tensor<1, dim> phi_j =
                              CrossProduct::trace_tangential(
                                fe_face_values[vec[block_index_j]].value(
                                  j, q_point),
                                normal[q_point]);

                            if (block_index_i == 0)
                              {
                                robin -= phi_i[q_point] * phi_j *
                                         fe_face_values.JxW(q_point);
                              }
                            else
                              { // if ( block_index_i == 1 )
                                robin += phi_i[q_point] * phi_j *
                                         fe_face_values.JxW(q_point);
                              }

                          } // rof: q_point

                        // Works for Complex and real, if the refractive index
                        // has a complex part, we also write something into the
                        // diagonal blocks
                        if (block_index_i == block_index_j)
                          cell_matrix(i, j) += omega.imag() * robin;
                        else if (block_index_i != block_index_j)
                          cell_matrix(i, j) += omega.real() * robin;

                      } // rof: dof_j

                  } // rof: dof_i

              } // fi: Robin boundary condition

            // --- interface condition ---
            if (cell->face(face)->boundary_id() >= 2)
              {
                // compute the normal
                std::vector<Tensor<1, dim>> normal(n_face_q_points);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  {
                    normal[q_point] = fe_face_values.normal_vector(q_point);
                  }

                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;

                    // we only want to compute this once
                    std::vector<Tensor<1, dim>> phi_i(n_face_q_points);
                    // std::vector<double>         surf_curl_phi_i(
                    // n_face_q_points ); std::vector<double>         div_phi_i(
                    // n_face_q_points );
                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        phi_i[q_point] = CrossProduct::trace_tangential(
                          fe_face_values[vec[block_index_i]].value(i, q_point),
                          normal[q_point]);

                        // For GIBC(alpha)
                        // surf_curl_phi_i[q_point] =
                        //  CrossProduct::curl(
                        //  CrossProduct::trace_tangential(fe_face_values[vec[block_index_i]].gradient(i,
                        //  q_point), normal[q_point]) ) * normal[q_point];

                        // div_phi_i[q_point] =
                        //   CrossProduct::div( CrossProduct::curl(
                        //   fe_face_values[vec[block_index_i]].hessian(i,
                        //   q_point) ) );

                      } // rof: q_point

                    for (const unsigned int j : fe_face_values.dof_indices())
                      {
                        const unsigned int block_index_j =
                          fe.system_to_block_index(j).first;

                        if (fe.has_support_on_face(i, face) == false ||
                            fe.has_support_on_face(j, face) == false)
                          continue;

                        double robin = 0;
                        // double interface_curl  = 0;  // for GIBC(alpha)
                        // double interface_div   = 0;  // for GIBC(alpha, beta)

                        for (unsigned int q_point = 0;
                             q_point < n_face_q_points;
                             q_point++)
                          {
                            Tensor<1, dim> phi_j =
                              CrossProduct::trace_tangential(
                                fe_face_values[vec[block_index_j]].value(
                                  j, q_point),
                                normal[q_point]);

                            // GIBC:
                            // double surf_curl_phi_j =
                            //  fe_face_values[vec[block_index_j]].curl(j,
                            //  q_point) * normal[q_point];

                            // double div_phi_j =
                            //   fe_face_values[vec[block_index_j]].divergence(j,
                            //   q_point);
                            //   //CrossProduct::div(CrossProduct::curl(fe_face_values[vec[block_index_j]].hessian(j,
                            //   q_point)));

                            // IBC ( = Robin):
                            robin += phi_i[q_point] * phi_j *
                                     fe_face_values.JxW(q_point);

                            // for GIBC:
                            // interface_curl +=
                            //  surf_curl_phi_i[q_point]
                            //  * surf_curl_phi_j
                            //  * fe_face_values.JxW(q_point);
                            //
                            // interface_div+=
                            //  div_phi_i[q_point]
                            //  * div_phi_j
                            //  * fe_face_values.JxW(q_point);

                          } // rof: q_point

                        if (block_index_i == block_index_j)
                          { // real values: (blocks: (0,0) and (1,1) )
                            cell_matrix(i, j) += omega.imag() * robin;
                            // cell_matrix(i, j) += beta.real()  *
                            // interface_curl; // for GIBC(alpha) cell_matrix(i,
                            // j) += alpha.real() * interface_div; // for
                            // GIBC(alpha, beta)
                          }
                        else if (block_index_i == 0 && block_index_j == 1)
                          {
                            cell_matrix(i, j) -= omega.real() * robin;
                            // cell_matrix(i, j) += beta.imag()  *
                            // interface_curl; // for GIBC(alpha) cell_matrix(i,
                            // j) += alpha.imag()  * interface_div; // for
                            // GIBC(alpha, beta)
                          }
                        else if (block_index_i == 1 && block_index_j == 0)
                          { // if ( block_index_i == 1 )
                            cell_matrix(i, j) += omega.real() * robin;
                            // cell_matrix(i, j) -= beta.imag() *
                            // interface_curl; // for GIBC(alpha, beta)
                            // cell_matrix(i, j) -= alpha.imag() *
                            // interface_div; // for GIBC(alpha, beta)
                          }

                      } // rof: dof_j

                  } // rof: dof_i

              } // fi: interface_condition

          } // rof: faces

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

      } // rof: active cell

    // synchronization between all processors
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    rhs_backup = system_rhs;

    pcout << " done!" << std::endl;
    timer.leave_subsection();
  }


  /*
   * assemble_rhs sets the right hand side of the equation
   *      curl ( curl ( E ) ) - w^2 E = f(x)
   * where f(x) is defined via the function "CurlRHS"
   * above.
   */
  template <int dim>
  void
  MaxwellProblem<dim>::assemble_system_rhs()
  {
    timer.enter_subsection("Assemble system right hand side");
    pcout << "assemble right hand side...";

    // choose the quadrature formulas
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    // set update flags
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // exact solution
    CurlRHS<dim> curl_rhs;

    Vector<double>              tmp(2 * dim);
    std::vector<Vector<double>> curl_rhs_values(n_q_points);
    for (unsigned int i = 0; i < n_q_points; i++)
      {
        curl_rhs_values[i] = tmp;
      }

    Tensor<1, dim> curl_rhs_tensor;

    const std::complex<double> imag_i(0.0, 1.0);

    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        // local right hand side
        Vector<double> cell_rhs(dofs_per_cell);

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int block_index_i =
              fe.system_to_block_index(i).first;
            const unsigned int pos = block_index_i * dim;

            curl_rhs.vector_value_list(fe_values.get_quadrature_points(),
                                       curl_rhs_values);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                curl_rhs_tensor[0] = curl_rhs_values[q_point][0 + pos];
                curl_rhs_tensor[1] = curl_rhs_values[q_point][1 + pos];
                if (dim == 3)
                  curl_rhs_tensor[2] = curl_rhs_values[q_point][2 + pos];

                cell_rhs[i] += fe_values[vec[block_index_i]].value(i, q_point) *
                               curl_rhs_tensor * fe_values.JxW(q_point);

              } // rof: q_point

          } // rof: dof_i

        for (const unsigned int i : fe_values.dof_indices())
          {
            system_rhs(local_dof_indices[i]) += cell_rhs[i];
          } // rof: dof_i

      } // rof: cell

    system_rhs.compress(VectorOperation::add);

    pcout << " done!" << std::endl;
    timer.leave_subsection();
  }


  template <int dim>
  void
  MaxwellProblem<dim>::update_interface_rhs()
  {
    timer.enter_subsection("update_interface");
    pcout << "update the interface...\t";

    // choose the quadrature formulas
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int curl_dim        = dim == 2 ? 1 : 3;

    // set update flags
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_gradients | update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;


    // Surface Operators:
    // s_tmp:
    std::vector<Tensor<1, dim, std::complex<double>>> update_value(
      n_face_q_points);
    std::vector<std::vector<std::vector<Tensor<1, dim, std::complex<double>>>>>
      s_value;
    {
      std::vector<std::vector<Tensor<1, dim, std::complex<double>>>> tmp;
      for (unsigned int k = 0; k < N_domains; k++)
        s_value.push_back(tmp);
    }

    // s_curl:
    std::vector<std::complex<double>> update_curl(n_face_q_points);
    std::vector<std::vector<std::vector<std::complex<double>>>> s_curl;
    {
      std::vector<std::vector<std::complex<double>>> tmp;
      for (unsigned int k = 0; k < N_domains; k++)
        s_curl.push_back(tmp);
    }


    // Material constants:
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        // cell dependent matirial constant
        // const double mu_term = 1.0;
        std::complex<double> omega = prm.get_wavenumber(cell->material_id());
        // const std::complex<double> beta(1.0 / (3.0 * omega * omega), 0.0);

        cell->get_dof_indices(local_dof_indices);


        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            unsigned int face_id = cell->face(face)->boundary_id();
            if (cell->face(face)->at_boundary() == false || face_id < 2)
              continue;

            fe_face_values.reinit(cell, face);

            // compute the normal
            std::vector<Tensor<1, dim>> normal(n_face_q_points);
            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++)
              {
                normal[q_point] = fe_face_values.normal_vector(q_point);
              }

            // face values:
            std::vector<Tensor<1, dim>> E_value_re(n_face_q_points);
            std::vector<Tensor<1, dim>> E_value_im(n_face_q_points);
            fe_face_values[E_re].get_function_values(solution, E_value_re);
            fe_face_values[E_im].get_function_values(solution, E_value_im);

            // face gradients:
            std::vector<Tensor<2, dim>> E_grad_re(n_face_q_points);
            std::vector<Tensor<2, dim>> E_grad_im(n_face_q_points);
            fe_face_values[E_re].get_function_gradients(solution, E_grad_re);
            fe_face_values[E_im].get_function_gradients(solution, E_grad_im);

            // face curls:
            std::vector<Tensor<1, curl_dim>> E_curl_re(n_face_q_points);
            std::vector<Tensor<1, curl_dim>> E_curl_im(n_face_q_points);
            fe_face_values[E_re].get_function_curls(solution, E_curl_re);
            fe_face_values[E_im].get_function_curls(solution, E_curl_im);

            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++)
              {
                update_value[q_point] =
                  imag_i * omega *
                  (CrossProduct::trace_tangential(E_value_re[q_point],
                                                  normal[q_point]) +
                   (imag_i * CrossProduct::trace_tangential(E_value_im[q_point],
                                                            normal[q_point])));

                update_value[q_point] +=
                  (trace(E_curl_re[q_point], normal[q_point]) +
                   (imag_i * trace(E_curl_im[q_point], normal[q_point])));

                // for GIBC(alpha):
                // update_curl[q_point] = beta  * (
                // CrossProduct::curl(CrossProduct::trace_tangential(E_grad_re[q_point],
                // normal[q_point])) * normal[q_point]
                //                + (imag_i *
                //                CrossProduct::curl(CrossProduct::trace_tangential(E_grad_im[q_point],
                //                normal[q_point])) * normal[q_point]) );
              }

            s_value[face_id - 2].push_back(update_value);
            s_curl[face_id - 2].push_back(update_curl);

          } // rof: face

      } // rof: cell

    // Write the data to the SurfaceOperator
    for (unsigned int face_id = 0; face_id < N_domains; face_id++)
      {
        SurfaceOperator.value(s_value[face_id], domain_id, face_id);
        SurfaceOperator.curl(s_curl[face_id], domain_id, face_id);
      }

    pcout << " done!" << std::endl;
    timer.leave_subsection();
  }

  template <int dim>
  void
  MaxwellProblem<dim>::assemble_interface_rhs()
  {
    timer.enter_subsection("assemble_interface_rhs");
    pcout << "assmable the interface right hand side...\t " << std::endl;

    system_rhs = 0;

    // choose the quadrature formulas
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;

    // set update flags
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    // SurfaceCommunicators:
    // g_tmp: (update_value)
    std::vector<Tensor<1, dim, std::complex<double>>> update_value(
      n_face_q_points);
    std::vector<std::vector<std::vector<Tensor<1, dim, std::complex<double>>>>>
      g_value;
    {
      std::vector<std::vector<Tensor<1, dim, std::complex<double>>>> tmp;
      for (unsigned int k = 0; k < N_domains; k++)
        g_value.push_back(tmp);
    }

    // g_tmp: (update_curl)
    std::vector<std::complex<double>> update_curl(n_face_q_points);
    std::vector<std::vector<std::vector<std::complex<double>>>> g_curl;
    {
      std::vector<std::vector<std::complex<double>>> tmp;
      for (unsigned int k = 0; k < N_domains; k++)
        g_curl.push_back(tmp);
    }

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<unsigned int> face_n(N_domains, 0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        // assamble the rhs of the interface condition
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            unsigned int face_id = cell->face(face)->boundary_id();
            if (cell->face(face)->at_boundary() == false || face_id < 2)
              continue;

            fe_face_values.reinit(cell, face);
            cell_rhs = 0;

            // compute the normal
            std::vector<Tensor<1, dim>> normal(n_face_q_points);
            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++)
              {
                normal[q_point] = fe_face_values.normal_vector(q_point);
              }


            for (const unsigned int i : fe_face_values.dof_indices())
              {
                if (fe.has_support_on_face(i, face) == false)
                  continue;

                const unsigned int block_index_i =
                  fe.system_to_block_index(i).first;

                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  {
                    update_value[q_point] = SurfaceOperator.value(
                      face_id - 2, domain_id)[face_n[face_id - 2]][q_point];
                    //+ S_curl.value(face_id - 2, domain_id)[face_n[face_id -
                    //2]][q_point];

                    // if(first_rhs) {
                    //   update_value[q_point] =  2.0 *
                    //   SurfaceOperator.value(face_id - 2,
                    //   domain_id)[face_n[face_id - 2]][q_point];
                    //  // update_curl[q_point]  =  2.0 *
                    //  SurfaceOperator.curl(face_id - 2,
                    //  domain_id)[face_n[face_id - 2]][q_point];
                    // } else {
                    //   update_value[q_point] = 2.0 *
                    //   SurfaceOperator.value(face_id - 2,
                    //   domain_id)[face_n[face_id - 2]][q_point]
                    //                         + g_out.value(face_id - 2,
                    //                         domain_id)[face_n[face_id -
                    //                         2]][q_point];

                    //  //update_curl[q_point]  = 2.0 *
                    //  SurfaceOperator.curl(face_id - 2,
                    //  domain_id)[face_n[face_id - 2]][q_point]
                    //  //                      - g_out.curl(face_id - 2,
                    //  domain_id)[face_n[face_id - 2]][q_point];
                    //}

                    Tensor<1, dim> phi_tangential_i =
                      CrossProduct::trace_tangential(
                        fe_face_values[vec[block_index_i]].value(i, q_point),
                        normal[q_point]);

                    // double surf_curl_phi_i =
                    //   fe_face_values[vec[block_index_i]].curl(i, q_point) *
                    //   normal[q_point];

                    std::complex<double> s_tmp =
                      //( (update_value[q_point] * phi_tangential_i) -
                      //(update_curl[q_point] * surf_curl_phi_i) ) *
                      //fe_face_values.JxW(q_point);
                      (update_value[q_point] * phi_tangential_i) *
                      fe_face_values.JxW(q_point);

                    if (block_index_i == 0)
                      {
                        cell_rhs[i] += s_tmp.real();
                      }
                    else
                      {
                        cell_rhs[i] += s_tmp.imag();
                      }

                  } // rof: q_point

                cell->get_dof_indices(local_dof_indices);
                system_rhs(local_dof_indices[i]) -= cell_rhs[i];

              } // rof: dof_i

            face_n[face_id - 2]++;
            // g_value[face_id - 2].push_back(update_value);
            // g_curl[face_id - 2].push_back(update_curl);
          } // rof: face

      } // rof: cell

    system_rhs += rhs_backup;
    system_rhs.compress(VectorOperation::add);

    // for( unsigned int face_id = 0; face_id < N_domains; face_id++) {
    //   g_out.value(g_value[face_id], domain_id, face_id);
    //   g_out.curl(g_curl[face_id], domain_id, face_id);
    // }

    if (first_rhs)
      first_rhs = false;

    pcout << "done!" << std::endl;
    timer.leave_subsection();
  }

  template <int dim>
  void
  MaxwellProblem<dim>::solve()
  {
    timer.enter_subsection("Solve");
    pcout << "solve the linear system with MUMPS...";

    SolverControl solver_control( // Alternative  SolverControl
      500,
      // 1e-6 * system_rhs.l2_norm()
      1e-4);

    TrilinosWrappers::SolverDirect::AdditionalData additional_data(
      false,         /* output_solver_details */
      "Amesos_Mumps" /* solver type */
      //"Amesos_Klu" /* solver type */
    );

    TrilinosWrappers::SolverDirect direct_solver(solver_control,
                                                 additional_data);

    TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs,
                                                       mpi_communicator);

    constraints.set_zero(distributed_solution);

    direct_solver.solve(system_matrix, distributed_solution, system_rhs);

    constraints.distribute(distributed_solution);

    solution = distributed_solution;

    solved = true;

    pcout << " done!" << std::endl;
    timer.leave_subsection();
  }


  template <int dim>
  void
  MaxwellProblem<dim>::output_results(const unsigned int step) const
  {
    pcout << "write the results...";

    // Define objects of our ComputeIntensity class
    ComputeIntensity<dim> intensities;
    DataOut<dim>          data_out;

    // and a DataOut object:
    data_out.attach_dof_handler(dof_handler);

    const std::string filename =
      prm.get_string("Output parameters", "Output file");
    const std::string format  = ".vtu";
    const std::string outfile = filename + "-step-" + std::to_string(step) +
                                "-domain-" + std::to_string(domain_id) + format;
    std::ofstream output(outfile);

    std::vector<std::string> solution_names;
    solution_names.emplace_back("Re_E1");
    solution_names.emplace_back("Re_E2");
    if (dim == 3)
      {
        solution_names.emplace_back("Re_E3");
      }
    solution_names.emplace_back("Im_E1");
    solution_names.emplace_back("Im_E2");
    if (dim == 3)
      {
        solution_names.emplace_back("Im_E3");
      }

    data_out.add_data_vector(solution, solution_names);
    data_out.add_data_vector(solution, intensities);

    data_out.build_patches();

    //  data_out.write_vtu_in_parallel(outfile.c_str(), mpi_communicator);
    data_out.write_vtu(output);

    pcout << " done!" << std::endl;
  }

  // === interpolate function ===
  // Interpolate the old solution to the new grid (since the solution is needed
  // for the DDM)
  template <int dim>
  void
  MaxwellProblem<dim>::refine()
  {
    pcout << "Start interpolation ... \t";

    // === Refinement ===
    // prepare the triangulation for refinement,
    triangulation.prepare_coarsening_and_refinement();

    // actually execute the refinement,
    triangulation.execute_coarsening_and_refinement();

    first_rhs = true;
    solved    = false;
    rebuild   = true;

    pcout << "done!" << std::endl;
  }

  // === interpolate function ===
  // Interpolate the old solution to the new grid (since the solution is needed
  // for the DDM)
  template <int dim>
  void
  MaxwellProblem<dim>::interpolate()
  {
    pcout << "Start interpolation ... \t";

    SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> soltrans(dof_handler);

    // === take a copy of the solution vector ===
    // Create initial indexsets pertaining to the grid before refinement

    // Transfer solution to vector that provides access to locally relevant DoFs
    TrilinosWrappers::MPI::Vector old_solution;
    old_solution.reinit(locally_owned_dofs,
                        locally_relevant_dofs,
                        mpi_communicator);
    old_solution = solution;

    // === Refinement ===
    // prepare the triangulation for refinement,
    triangulation.prepare_coarsening_and_refinement();

    // tell the SolutionTransfer object that we intend to do pure refinement,
    soltrans.prepare_for_coarsening_and_refinement(old_solution);

    // actually execute the refinement,
    triangulation.execute_coarsening_and_refinement();

    // and redistribute dofs.
    dof_handler.distribute_dofs(fe);

    // === Aftermath ===
    // Recreate locally_owned_dofs and locally_relevant_dofs index sets
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Constraits
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      0 /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      dim /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    constraints.close();

    // transfer the solution to the new grid
    solution.reinit(locally_owned_dofs, mpi_communicator);
    soltrans.interpolate(old_solution, solution);
    constraints.distribute(solution);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    rhs_backup.reinit(locally_owned_dofs, mpi_communicator);

    // create sparsity pattern:
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    // create the system matrix
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);

    rebuild = true;
    solved  = false;

    TrilinosWrappers::MPI::Vector tmp = solution;
    assemble_system();
    solution = tmp;

    pcout << "done!" << std::endl;
  }

  // === interpolate function ===
  // Interpolate the old solution to the new grid (since the solution is needed
  // for the DDM)
  template <int dim>
  void
  MaxwellProblem<dim>::interpolate_global()
  {
    pcout << "Start interpolation ... \t";

    SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> soltrans(dof_handler);

    // === take a copy of the solution vector ===
    // Create initial indexsets pertaining to the grid before refinement

    // Transfer solution to vector that provides access to locally relevant DoFs
    TrilinosWrappers::MPI::Vector old_solution;
    old_solution.reinit(locally_owned_dofs,
                        locally_relevant_dofs,
                        mpi_communicator);
    old_solution = solution;

    // === Refinement ===
    // prepare the triangulation for refinement,
    triangulation.set_all_refine_flags();
    triangulation.prepare_coarsening_and_refinement();

    // tell the SolutionTransfer object that we intend to do pure refinement,
    soltrans.prepare_for_coarsening_and_refinement(old_solution);

    // actually execute the refinement,
    triangulation.execute_coarsening_and_refinement();

    // and redistribute dofs.
    dof_handler.distribute_dofs(fe);

    // === Aftermath ===
    // Recreate locally_owned_dofs and locally_relevant_dofs index sets
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Constraits
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      0 /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      dim /* vector component*/,
      DirichletBoundaryValues<dim>(),
      1 /* boundary id*/,
      constraints);

    constraints.close();

    // transfer the solution to the new grid
    solution.reinit(locally_owned_dofs, mpi_communicator);
    soltrans.interpolate(old_solution, solution);
    constraints.distribute(solution);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    rhs_backup.reinit(locally_owned_dofs, mpi_communicator);

    // create sparsity pattern:
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    // create the system matrix
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);

    rebuild = true;
    solved  = false;

    TrilinosWrappers::MPI::Vector tmp = solution;
    assemble_system();
    solution = tmp;

    pcout << "Done!" << std::endl;
  }

  // === errpr estimator ===
  template <int dim>
  void
  MaxwellProblem<dim>::error_estimator(Vector<float> &error_indicators) const
  {
    // choose the quadrature formulas
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // set update flags
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | update_hessians |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_gradients |
                                       update_normal_vectors |
                                       update_JxW_values);

    // define some constants:
    const unsigned int curl_dim = (dim == 2) ? 1 : 3;

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    // cell:
    // cell values:
    std::vector<Tensor<1, dim>> E_value_re(n_q_points), E_value_im(n_q_points);
    // std::vector<Tensor<1,dim>> s_value_re(n_q_points),
    // s_value_im(n_q_points);
    //  cell divs:
    std::vector<double> E_div_re(n_q_points), E_div_im(n_q_points);
    // cell hessians:
    std::vector<Tensor<3, dim>> E_hessian_re(n_q_points),
      E_hessian_im(n_q_points);
    // face:
    // face values:
    std::vector<Tensor<1, dim>> E_face_value_re(n_face_q_points),
      E_face_value_im(n_face_q_points);
    std::vector<Tensor<1, dim>> s_face_value_re(n_face_q_points),
      s_face_value_im(n_face_q_points);
    // face curls:
    std::vector<Tensor<1, curl_dim>> E_face_curl_re(n_face_q_points),
      E_face_curl_im(n_face_q_points);
    // neighbor face
    // neighbor face values:
    std::vector<Tensor<1, dim>> neighbor_value_re(n_face_q_points),
      neighbor_value_im(n_face_q_points);
    // neighbor face curls:
    std::vector<Tensor<1, curl_dim>> neighbor_curl_re(n_face_q_points),
      neighbor_curl_im(n_face_q_points);
    // neighbour rhs values:
    std::vector<Tensor<1, dim>> neighbor_rhs_re(n_face_q_points),
      neighbor_rhs_im(n_face_q_points);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          {
            continue;
          }

        fe_values.reinit(cell);

        // get the material values
        const double               mu_term = 1.0;
        const std::complex<double> omega =
          prm.get_wavenumber(cell->material_id());

        // TODO ONLY WORKS WITH EITHER REAL OR IMAGINARY;
        const double kappa_term = std::pow(omega, 2).real();

        // get the values form the cell:
        fe_values[E_re].get_function_values(solution, E_value_re);
        fe_values[E_im].get_function_values(solution, E_value_im);
        // fe_values[E_re].get_function_values(system_rhs, s_value_re);
        // fe_values[E_im].get_function_values(system_rhs, s_value_im);

        // get the hessian_matrix (needed to compute curl crul (u)):
        fe_values[E_re].get_function_hessians(solution, E_hessian_re);
        fe_values[E_im].get_function_hessians(solution, E_hessian_im);

        // get the divergence from the cell:
        fe_values[E_re].get_function_divergences(solution, E_div_re);
        fe_values[E_im].get_function_divergences(solution, E_div_im);
        double cell_error = 0.0;
        double eta_RK     = 0.0;
        double eta_JK     = 0.0;

        // compute the cell based residual term:
        // eta^2_{R,K} = h^2_K/p^2 ( ||curl(mu^{-1} * curl(E_h)) - epsilon *
        // omega^2 E_h - s ||^2 + || div(epsilon * omega^2 E_h) ||^2 )
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            auto E_curl_curl =
              CrossProduct::curl_curl(E_hessian_re[q_point]) +
              imag_i * CrossProduct::curl_curl(E_hessian_im[q_point]);
            auto E_value = E_value_re[q_point] + imag_i * E_value_im[q_point];
            // auto s_value      = s_value_re[q_point] + imag_i *
            // s_value_im[q_point];      // with right hand side auto tmp_1 =
            // s_value - ( mu_term * E_curl_curl ) + ( kappa_term * E_value );
            // // with right hand side
            auto tmp_1 = -(mu_term * E_curl_curl) + (kappa_term * E_value);
            eta_RK += std::abs(tmp_1 * tmp_1) * fe_values.JxW(q_point);
            auto tmp_2 =
              kappa_term * (E_div_re[q_point] + imag_i * E_div_im[q_point]);
            eta_RK += std::abs(tmp_2 * tmp_2) * fe_values.JxW(q_point);
          }

        // compute the face based term:
        // eta^2_{J.K} = 1/2 Sum_{faces} ( h_f/p || [ Trace( mu^{-1} * curl(E_h)
        // ) ] ||^2 + || [ n_f * epsilon \omega \trace(E_h) + \trace(s) ] ||^2 )
        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          {
            /*
             * First, if this face is part of the boundary, then there is
             * nothing to do. However, to make things easier when summing up the
             * contributions of the faces of cells, we enter this face into the
             * list of faces with a zero contribution to the error.
             */
            if (cell->face(face_no)->at_boundary())
              {
                continue;
              }
            /*
             * First, if the neighboring cell is on the same level as this one,
             * i.e. neither further refined not coarser, then the one with the
             * lower index within this level does the work. In other words: if
             * the other one has a lower index, then skip work on this face:
             */
            if ((cell->neighbor(face_no)->has_children() == false) &&
                (cell->neighbor(face_no)->level() == cell->level()) &&
                (cell->neighbor(face_no)->index() < cell->index()))
              {
                continue;
              }
            /*
             * Likewise, we always work from the coarser cell if this and its
             * neighbor differ in refinement. Thus, if the neighboring cell is
             * less refined than the present one, then do nothing since we
             * integrate over the subfaces when we visit the coarse cell.
             */
            if (cell->at_boundary(face_no) == false)
              {
                if (cell->neighbor(face_no)->level() < cell->level())
                  {
                    continue;
                  }
              }

            fe_face_values.reinit(cell, face_no);
            // get the normal:
            std::vector<Tensor<1, dim>> normal(n_face_q_points);
            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++)
              {
                normal[q_point] = fe_face_values.normal_vector(q_point) *
                                  fe_face_values.JxW(q_point);
              }

            // get the curl:
            fe_face_values[E_re].get_function_curls(solution, E_face_curl_re);
            fe_face_values[E_im].get_function_curls(solution, E_face_curl_im);
            //_face get the value:
            fe_face_values[E_re].get_function_values(solution, E_face_value_re);
            fe_face_values[E_im].get_function_values(solution, E_face_value_im);
            //_face get the value:
            fe_face_values[E_re].get_function_values(solution, s_face_value_re);
            fe_face_values[E_im].get_function_values(solution, s_face_value_im);
            //_face get the value:
            fe_face_values[E_re].get_function_values(solution, s_face_value_re);
            fe_face_values[E_im].get_function_values(solution, s_face_value_im);

            // differentiate between regular and irregular neighbour cells
            if (cell->face(face_no)->has_children() == false)
              {
                // integrate over regular nighbour cell:
                Assert(cell->neighbor(face_no).state() == IteratorState::valid,
                       ExcInternalError());

                // get the neighbour cell:
                const auto         neighbor = cell->neighbor(face_no);
                const unsigned int neighbor_neighbor =
                  cell->neighbor_of_neighbor(face_no);
                fe_face_values.reinit(neighbor, neighbor_neighbor);

                // get the value of the neighbor
                fe_face_values[E_re].get_function_values(solution,
                                                         neighbor_value_re);
                fe_face_values[E_im].get_function_values(solution,
                                                         neighbor_value_im);

                // face_get the curl of the neighbor
                fe_face_values[E_re].get_function_curls(solution,
                                                        neighbor_curl_re);
                fe_face_values[E_im].get_function_curls(solution,
                                                        neighbor_curl_im);

                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  {
                    auto E_jump_curl = (E_face_curl_re[q_point] +
                                        imag_i * E_face_curl_im[q_point]) -
                                       (neighbor_curl_re[q_point] +
                                        imag_i * neighbor_curl_im[q_point]);

                    auto tmp_1 = mu_term * CrossProduct::trace(E_jump_curl,
                                                               normal[q_point]);
                    eta_JK += (cell->face(face_no)->diameter() /
                               (2.0 * (fe.degree + 1) * std::sqrt(1))) *
                              std::abs(tmp_1 * tmp_1) *
                              fe_face_values.JxW(q_point);

                  } // rof: q_point
              }
            else
              {
                // integrate over irregular nighbour cell:
                Assert(cell->neighbor(face_no).state() == IteratorState::valid,
                       ExcInternalError());

                const typename DoFHandler<dim>::face_iterator face =
                  cell->face(face_no);

                const unsigned int neighbor_neighbor =
                  cell->neighbor_of_neighbor(face_no);

                for (unsigned int subface_no = 0;
                     subface_no < face->n_children();
                     ++subface_no)
                  {
                    const auto neighbor_child =
                      cell->neighbor_child_on_subface(face_no, subface_no);

                    Assert(neighbor_child->face(neighbor_neighbor) ==
                             cell->face(face_no)->child(subface_no),
                           ExcInternalError());
                    fe_face_values.reinit(neighbor_child, neighbor_neighbor);

                    // get the value of the neighbor
                    fe_face_values[E_re].get_function_values(solution,
                                                             neighbor_value_re);
                    fe_face_values[E_im].get_function_values(solution,
                                                             neighbor_value_im);

                    // face_get the curl of the neighbor
                    fe_face_values[E_re].get_function_curls(solution,
                                                            neighbor_curl_re);
                    fe_face_values[E_im].get_function_curls(solution,
                                                            neighbor_curl_im);

                    // get the rhs values:
                    fe_face_values[E_re].get_function_values(system_rhs,
                                                             neighbor_rhs_re);
                    fe_face_values[E_im].get_function_values(system_rhs,
                                                             neighbor_rhs_im);

                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        auto E_jump_curl = (E_face_curl_re[q_point] +
                                            imag_i * E_face_curl_im[q_point]) -
                                           (neighbor_curl_re[q_point] +
                                            imag_i * neighbor_curl_im[q_point]);

                        auto tmp_1 =
                          mu_term *
                          CrossProduct::trace(E_jump_curl, normal[q_point]);
                        eta_JK += (cell->face(face_no)->diameter() /
                                   (2.0 * (fe.degree + 1))) *
                                  std::abs(tmp_1 * tmp_1) *
                                  fe_face_values.JxW(q_point);

                        auto E_jump = (E_face_value_re[q_point] +
                                       (imag_i * E_face_value_im[q_point]) -
                                       s_face_value_re[q_point] -
                                       (imag_i * s_face_value_im[q_point])) -
                                      (neighbor_value_re[q_point] +
                                       (imag_i * neighbor_value_im[q_point]) -
                                       neighbor_value_re[q_point] -
                                       (imag_i * neighbor_value_im[q_point]));

                        auto tmp_2 =
                          kappa_term *
                          CrossProduct::trace_tangential(E_jump,
                                                         normal[q_point]);
                        eta_JK += (cell->face(face_no)->diameter() /
                                   (2.0 * (fe.degree + 1))) *
                                  std::abs(tmp_2 * tmp_2) *
                                  fe_face_values.JxW(q_point);

                      } // rof: q_point

                  } // rof: sub face

              } // else

          } // rof: face_no

        eta_RK *=
          std::pow((cell->diameter() / std::sqrt(2)) / (fe.degree + 1.0), 2);
        cell_error = std::sqrt(eta_RK + eta_JK);
        error_indicators(cell->active_cell_index()) += cell_error;

      } // rof: cell

    error_indicators.compress(VectorOperation::add);
  }

  template <int dim>
  void
  MaxwellProblem<dim>::mark_for_refinement_error_estimator()
  {
    // only refine when the system was solved previously
    // so we don't refine the grid in the first call
    if (solved)
      {
        // LinearAlgebra::Vector<float> error_indicators;
        Vector<float> error_indicators(triangulation.n_active_cells());

        // compute the error estimator:
        error_estimator(error_indicators);

        // parallel::shared::GridRefinement::refine_and_coarsen_fixed_number(
        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        error_indicators,
                                                        0.40,
                                                        0.00);

        // prepare the triangulation for refinement,
        triangulation.prepare_coarsening_and_refinement();

        // Print the current error
        const double L2_error = VectorTools::compute_global_error(
          triangulation, error_indicators, VectorTools::L2_norm
          // VectorTools::H1_seminorm
          // VectorTools::Linfty_norm
        );
        pcout << "L2 error: " << L2_error << std::endl;
      }
  }

  template <int dim>
  void
  MaxwellProblem<dim>::prepare_mark_interface_for_refinement()
  {
    // Refinement Operator
    std::vector<std::vector<bool>> r_coarsen(N_domains);
    std::vector<std::vector<bool>> r_refinement(N_domains);

    // loop over all cells:
    for (auto &cell : dof_handler.active_cell_iterators())
      {
        // skip all non-locally owned cells
        if (!cell->is_locally_owned())
          continue;

        // loop over all faces:
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            // get the face_id
            unsigned int face_id = cell->face(face)->boundary_id();

            // skip all faces, that are not located at the boundary
            if (!cell->face(face)->at_boundary())
              continue;

            // skip all faces with a face_id < 2 ( as they do not belong to
            // interfaces)
            if (face_id < 2)
              continue;

            r_coarsen[face_id - 2].push_back(cell->coarsen_flag_set());
            r_refinement[face_id - 2].push_back(cell->refine_flag_set());
          }

      } // rof: cell

    // Write the data to the RefinementOperator
    for (unsigned int face_id = 0; face_id < N_domains; face_id++)
      {
        RefinementOperator.coarsen(r_coarsen[face_id], domain_id, face_id);
        RefinementOperator.refinement(r_refinement[face_id],
                                      domain_id,
                                      face_id);
      }
  }

  template <int dim>
  void
  MaxwellProblem<dim>::apply_mark_interface_for_refinement()
  {
    pcout << "Mark the interface for refinement... ";

    // counter
    std::vector<unsigned int> face_n(N_domains, 0);

    // loop over all cells
    for (auto &cell : dof_handler.active_cell_iterators())
      {
        // skip all non-locally owned cells
        if (!cell->is_locally_owned())
          continue;

        // loop over all faces:
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            // get the face_id
            unsigned int face_id = cell->face(face)->boundary_id();

            // skip all faces, that are not located at the boundary
            if (!cell->face(face)->at_boundary())
              continue;

            // skip all faces with a face_id < 2 ( as they do not belong to
            // interfaces)
            if (face_id < 2)
              continue;

            // we need to priorize one domain, here we choose the domain with
            // the higher domain_id to define the refinement level, therefore we
            // skip this face if the domain_id is higher than the domain_id of
            // its neighbor
            // if (domain_id > face_id - 2)
            //  continue;

            // first remove any existing refine or coarsen flags
            // cell->clear_refine_flag();
            // cell->clear_coarsen_flag();

            if (RefinementOperator.refinement(face_id - 2,
                                              domain_id)[face_n[face_id - 2]])
              {
                cell->set_refine_flag();
              }
            else if (RefinementOperator.coarsen(face_id - 2,
                                                domain_id)[face_n[face_id - 2]])
              {
                // cell->set_coarsen_flag();
              }

            face_n[face_id - 2]++;
          } // rof: face

      } // rof: cell

    pcout << " done!" << std::endl;
  }

  // === public functions ===
  template <int dim>
  void
  MaxwellProblem<dim>::initialize()
  {
    setup_system();
    assemble_system();
    // assemble_rhs();
  }

  template <int dim>
  void
  MaxwellProblem<dim>::print_results(const unsigned int step) const
  {
    output_results(step);
    timer.print_summary();
  }

  // return functions:
  template <int dim>
  Triangulation<dim> &
  MaxwellProblem<dim>::return_triangulation()
  {
    return triangulation;
  }


  template <int dim>
  SurfaceCommunicator<dim>
  MaxwellProblem<dim>::return_g_in()
  {
    return SurfaceOperator;
  }

  template <int dim>
  SurfaceCommunicator<dim>
  MaxwellProblem<dim>::return_g_out()
  {
    return g_out;
  }

  template <int dim>
  RefinementCommunicator<dim>
  MaxwellProblem<dim>::return_refine()
  {
    return RefinementOperator;
  }


  // update
  template <int dim>
  void
  MaxwellProblem<dim>::update_g_out(SurfaceCommunicator<dim> g)
  {
    g_out = g;
  }

  template <int dim>
  void
  MaxwellProblem<dim>::update_g_in(SurfaceCommunicator<dim> g)
  {
    SurfaceOperator = g;
  }

  template <int dim>
  void
  MaxwellProblem<dim>::update_refine(RefinementCommunicator<dim> r)
  {
    RefinementOperator = r;
  }


  // compile the tamplate with certain parameters
  template class MaxwellProblem<2>;
  template class MaxwellProblem<3>;

} // namespace KirasFM
