#include <iostream>
#include <fstream>
#include <vector>
#include <netcdf>
#include <mpi.h>

void output(double*** state, double etime) {
    int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid;
    int i, k;
    static int num_out = 0;
    MPI_Offset len, st1[1], ct1[1], st3[3], ct3[3];

    // Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
    double** dens = new double*[nx];
    double** uwnd = new double*[nx];
    double** wwnd = new double*[nx];
    double** theta = new double*[nx];
    for (i = 0; i < nx; i++) {
        dens[i] = new double[nz];
        uwnd[i] = new double[nz];
        wwnd[i] = new double[nz];
        theta[i] = new double[nz];
    }
    double etimearr[1];

    // Inform the user
    if (mainproc) std::cout << "*** OUTPUT ***" << std::endl;

    // If the elapsed time is zero, create the file. Otherwise, open the file
    if (etime == 0) {
        // Create the file
        nc_create_par("output.nc", NC_NETCDF4 | NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid);
        // Create the dimensions
        len = NC_UNLIMITED; nc_def_dim(ncid, "t", len, &t_dimid);
        len = nx_glob; nc_def_dim(ncid, "x", len, &x_dimid);
        len = nz_glob; nc_def_dim(ncid, "z", len, &z_dimid);
        // Create the variables
#ifdef SINGLE_PREC
        nc_def_var(ncid, "t", NC_FLOAT, 1, &t_dimid, &t_varid);
        nc_def_var(ncid, "dens", NC_FLOAT, 3, &dens_varid);
        nc_def_var(ncid, "uwnd", NC_FLOAT, 3, &uwnd_varid);
        nc_def_var(ncid, "wwnd", NC_FLOAT, 3, &wwnd_varid);
        nc_def_var(ncid, "theta", NC_FLOAT, 3, &theta_varid);
#else
        nc_def_var(ncid, "t", NC_DOUBLE, 1, &t_dimid, &t_varid);
        nc_def_var(ncid, "dens", NC_DOUBLE, 3, &dens_varid);
        nc_def_var(ncid, "uwnd", NC_DOUBLE, 3, &uwnd_varid);
        nc_def_var(ncid, "wwnd", NC_DOUBLE, 3, &wwnd_varid);
        nc_def_var(ncid, "theta", NC_DOUBLE, 3, &theta_varid);
#endif
        // End "define" mode
        nc_enddef(ncid);
    } else {
        // Open the file
        nc_open_par("output.nc", NC_WRITE, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid);
        // Get the variable IDs
        nc_inq_varid(ncid, "dens", &dens_varid);
        nc_inq_varid(ncid, "uwnd", &uwnd_varid);
        nc_inq_varid(ncid, "wwnd", &wwnd_varid);
        nc_inq_varid(ncid, "theta", &theta_varid);
        nc_inq_varid(ncid, "t", &t_varid);
    }

    // Store perturbed values in the temp arrays for output
    for (k = 0; k < nz; k++) {
        for (i = 0; i < nx; i++) {
            dens[i][k] = state[i][k][ID_DENS];
            uwnd[i][k] = state[i][k][ID_UMOM] / (hy_dens_cell[k] + state[i][k][ID_DENS]);
            wwnd[i][k] = state[i][k][ID_WMOM] / (hy_dens_cell[k] + state[i][k][ID_DENS]);
            theta[i][k] = (state[i][k][ID_RHOT] + hy_dens_theta_cell[k]) / (hy_dens_cell[k] + state[i][k][ID_DENS]) - hy_dens_theta_cell[k] / hy_dens_cell[k];
        }
    }

    // Write the grid data to file with all the processes writing collectively
#ifdef SINGLE_PREC
    st3[0] = i_beg; st3[1] = k_beg; st3[2] = num_out + 1;
    ct3[0] = nx; ct3[1] = nz; ct3[2] = 1;
    nc_put_vara_float_all(ncid, dens_varid, st3, ct3, &dens[0][0]);
    nc_put_vara_float_all(ncid, uwnd_varid, st3, ct3, &uwnd[0][0]);
    nc_put_vara_float_all(ncid, wwnd_varid, st3, ct3, &wwnd[0][0]);
    nc_put_vara_float_all(ncid, theta_varid, st3, ct3, &theta[0][0]);
#else
    st3[0] = i_beg; st3[1] = k_beg; st3[2] = num_out + 1;
    ct3[0] = nx; ct3[1] = nz; ct3[2] = 1;
    nc_put_vara_double_all(ncid, dens_varid, st3, ct3, &dens[0][0]);
    nc_put_vara_double_all(ncid, uwnd_varid, st3, ct3, &uwnd[0][0]);
    nc_put_vara_double_all(ncid, wwnd_varid, st3, ct3, &wwnd[0][0]);
    nc_put_vara_double_all(ncid, theta_varid, st3, ct3, &theta[0][0]);
#endif

    // Only the main process needs to write the elapsed time
    // Begin "independent" write mode
    nc_begin_indep_data(ncid);
    // Write elapsed time to file
    if (mainproc) {
#ifdef SINGLE_PREC
        st1[0] = num_out + 1; ct1[0] = 1;
        etimearr[0] = etime;
        nc_put_vara_float(ncid, t_varid, st1, ct1, etimearr);
#else
        st1[0] = num_out + 1; ct1[0] = 1;
        etimearr[0] = etime;
        nc_put_vara_double(ncid, t_varid, st1, ct1, etimearr);
#endif
    }
    // End "independent" write mode
    nc_end_indep_data(ncid);

    // Close the file
    nc_close(ncid);

    // Increment the number of outputs
    num_out++;

    // Deallocate the temp arrays
    for (i = 0; i < nx; i++) {
        delete[] dens[i];
        delete[] uwnd[i];
        delete[] wwnd[i];
        delete[] theta[i];
    }
    delete[] dens;
    delete[] uwnd;
    delete[] wwnd;
    delete[] theta;
}
