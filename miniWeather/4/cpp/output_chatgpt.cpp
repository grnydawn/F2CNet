#include <netcdf.h>
#include <netcdf_par.h>
#include <vector>
#include <iostream>
#include <mpi.h>

extern bool mainproc; // Assuming this is defined somewhere
extern int hs, nx, nz, NUM_VARS, nx_glob, nz_glob, i_beg, k_beg;
extern std::vector<std::vector<std::vector<double>>> state;
extern std::vector<double> hy_dens_cell, hy_dens_theta_cell;

void ncwrap(int ierr, int line) {
    if (ierr != NC_NOERR) {
        std::cerr << "NetCDF Error at line: " << line << std::endl;
        std::cerr << nc_strerror(ierr) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }
}

void output(const std::vector<std::vector<std::vector<double>>>& state, double etime) {
    int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid;
    static int num_out = 0;
    MPI_Offset len, st1[1], ct1[1], st3[3], ct3[3];
    std::vector<std::vector<double>> dens(nx, std::vector<double>(nz));
    std::vector<std::vector<double>> uwnd(nx, std::vector<double>(nz));
    std::vector<std::vector<double>> wwnd(nx, std::vector<double>(nz));
    std::vector<std::vector<double>> theta(nx, std::vector<double>(nz));
    double etimearr[1];

    if (mainproc) std::cout << "*** OUTPUT ***" << std::endl;

    // If the elapsed time is zero, create the file. Otherwise, open the file
    if (etime == 0) {
        // Create the file
        ncwrap(nc_create_par("output.nc", NC_NETCDF4 | NC_MPIIO | NC_CLOBBER, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid), __LINE__);

        // Create the dimensions
        len = NC_UNLIMITED;
        ncwrap(nc_def_dim(ncid, "t", len, &t_dimid), __LINE__);
        len = nx_glob;
        ncwrap(nc_def_dim(ncid, "x", len, &x_dimid), __LINE__);
        len = nz_glob;
        ncwrap(nc_def_dim(ncid, "z", len, &z_dimid), __LINE__);

        // Create the variables
        #ifdef SINGLE_PREC
        ncwrap(nc_def_var(ncid, "t", NC_FLOAT, 1, &t_dimid, &t_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "dens", NC_FLOAT, 3, (int[]){x_dimid, z_dimid, t_dimid}, &dens_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "uwnd", NC_FLOAT, 3, (int[]){x_dimid, z_dimid, t_dimid}, &uwnd_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "wwnd", NC_FLOAT, 3, (int[]){x_dimid, z_dimid, t_dimid}, &wwnd_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "theta", NC_FLOAT, 3, (int[]){x_dimid, z_dimid, t_dimid}, &theta_varid), __LINE__);
        #else
        ncwrap(nc_def_var(ncid, "t", NC_DOUBLE, 1, &t_dimid, &t_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "dens", NC_DOUBLE, 3, (int[]){x_dimid, z_dimid, t_dimid}, &dens_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "uwnd", NC_DOUBLE, 3, (int[]){x_dimid, z_dimid, t_dimid}, &uwnd_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "wwnd", NC_DOUBLE, 3, (int[]){x_dimid, z_dimid, t_dimid}, &wwnd_varid), __LINE__);
        ncwrap(nc_def_var(ncid, "theta", NC_DOUBLE, 3, (int[]){x_dimid, z_dimid, t_dimid}, &theta_varid), __LINE__);
        #endif

        // End "define" mode
        ncwrap(nc_enddef(ncid), __LINE__);
    } else {
        // Open the file
        ncwrap(nc_open_par("output.nc", NC_WRITE | NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid), __LINE__);

        // Get the variable IDs
        ncwrap(nc_inq_varid(ncid, "dens", &dens_varid), __LINE__);
        ncwrap(nc_inq_varid(ncid, "uwnd", &uwnd_varid), __LINE__);
        ncwrap(nc_inq_varid(ncid, "wwnd", &wwnd_varid), __LINE__);
        ncwrap(nc_inq_varid(ncid, "theta", &theta_varid), __LINE__);
        ncwrap(nc_inq_varid(ncid, "t", &t_varid), __LINE__);
    }

    // Store perturbed values in the temp arrays for output
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
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
    ncwrap(nc_put_vara_float_all(ncid, dens_varid, st3, ct3, &dens[0][0]), __LINE__);
    ncwrap(nc_put_vara_float_all(ncid, uwnd_varid, st3, ct3, &uwnd[0][0]), __LINE__);
    ncwrap(nc_put_vara_float_all(ncid, wwnd_varid, st3, ct3, &wwnd[0][0]), __LINE__);
    ncwrap(nc_put_vara_float_all(ncid, theta_varid, st3, ct3, &theta[0][0]), __LINE__);
    #else
    st3[0] = i_beg; st3[1] = k_beg; st3[2] = num_out + 1;
    ct3[0] = nx; ct3[1] = nz; ct3[2] = 1;
    ncwrap(nc_put_vara_double_all(ncid, dens_varid, st3, ct3, &dens[0][0]), __LINE__);
    ncwrap(nc_put_vara_double_all(ncid, uwnd_varid, st3, ct3, &uwnd[0][0]), __LINE__);
    ncwrap(nc_put_vara_double_all(ncid, wwnd_varid, st3, ct3, &wwnd[0][0]), __LINE__);
    ncwrap(nc_put_vara_double_all(ncid, theta_varid, st3, ct3, &theta[0][0]), __LINE__);
    #endif

    // Only the main process needs to write the elapsed time
    // Begin "independent" write mode
    ncwrap(nc_var_par_access(ncid, t_varid, NC_INDEPENDENT), __LINE__);
    // Write elapsed time to file
    if (mainproc) {
        #ifdef SINGLE_PREC
        st1[0] = num_out + 1; ct1[0] = 1;
        etimearr[0] = etime;
        ncwrap(nc_put_vara_float(ncid, t_varid, st1, ct1, etimearr), __LINE__);
        #else
        st1[0] = num_out + 1; ct1[0] = 1;
        etimearr[0] = etime;
        ncwrap(nc_put_vara_double(ncid, t_varid, st1, ct1, etimearr), __LINE__);
        #endif
    }
    // End "independent" write mode
    ncwrap(nc_var_par_access(ncid, t_varid, NC_COLLECTIVE), __LINE__);

    // Close the file
    ncwrap(nc_close(ncid), __LINE__);

    // Increment the number of outputs
    num_out++;

    // Deallocate the temp arrays (automatic deallocation happens in C++ with std::vector)
}

