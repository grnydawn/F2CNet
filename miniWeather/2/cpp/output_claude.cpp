#include <vector>
#include <iostream>
#include <pnetcdf.h>
#include <mpi.h>

void output(const std::vector<std::vector<std::vector<real>>>& state, real etime) {
    int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid;
    static int num_out = 0;
    MPI_Offset len, st1[1], ct1[1], st3[3], ct3[3];
    std::vector<real> etimearr(1);

    // Inform the user
    if (mainproc) std::cout << "*** OUTPUT ***" << std::endl;

    // Allocate temporary arrays
    std::vector<std::vector<real>> dens(nx, std::vector<real>(nz));
    std::vector<std::vector<real>> uwnd(nx, std::vector<real>(nz));
    std::vector<std::vector<real>> wwnd(nx, std::vector<real>(nz));
    std::vector<std::vector<real>> theta(nx, std::vector<real>(nz));

    if (etime == 0) {
        // Create the file
        ncwrap(ncmpi_create(MPI_COMM_WORLD, "output.nc", NC_CLOBBER, MPI_INFO_NULL, &ncid), __LINE__);

        // Create dimensions
        len = NC_UNLIMITED; ncwrap(ncmpi_def_dim(ncid, "t", len, &t_dimid), __LINE__);
        len = nx_glob; ncwrap(ncmpi_def_dim(ncid, "x", len, &x_dimid), __LINE__);
        len = nz_glob; ncwrap(ncmpi_def_dim(ncid, "z", len, &z_dimid), __LINE__);

        // Create variables
        int dims[3] = {x_dimid, z_dimid, t_dimid};
        #ifdef SINGLE_PREC
            ncwrap(ncmpi_def_var(ncid, "t", NC_FLOAT, 1, &t_dimid, &t_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "dens", NC_FLOAT, 3, dims, &dens_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "uwnd", NC_FLOAT, 3, dims, &uwnd_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "wwnd", NC_FLOAT, 3, dims, &wwnd_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "theta", NC_FLOAT, 3, dims, &theta_varid), __LINE__);
        #else
            ncwrap(ncmpi_def_var(ncid, "t", NC_DOUBLE, 1, &t_dimid, &t_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "dens", NC_DOUBLE, 3, dims, &dens_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "uwnd", NC_DOUBLE, 3, dims, &uwnd_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "wwnd", NC_DOUBLE, 3, dims, &wwnd_varid), __LINE__);
            ncwrap(ncmpi_def_var(ncid, "theta", NC_DOUBLE, 3, dims, &theta_varid), __LINE__);
        #endif

        // End "define" mode
        ncwrap(ncmpi_enddef(ncid), __LINE__);
    } else {
        // Open the file
        ncwrap(ncmpi_open(MPI_COMM_WORLD, "output.nc", NC_WRITE, MPI_INFO_NULL, &ncid), __LINE__);

        // Get the variable IDs
        ncwrap(ncmpi_inq_varid(ncid, "dens", &dens_varid), __LINE__);
        ncwrap(ncmpi_inq_varid(ncid, "uwnd", &uwnd_varid), __LINE__);
        ncwrap(ncmpi_inq_varid(ncid, "wwnd", &wwnd_varid), __LINE__);
        ncwrap(ncmpi_inq_varid(ncid, "theta", &theta_varid), __LINE__);
        ncwrap(ncmpi_inq_varid(ncid, "t", &t_varid), __LINE__);
    }

    // Store perturbed values in the temp arrays for output
    for (int k = 1; k <= nz; k++) {
        for (int i = 1; i <= nx; i++) {
            dens[i-1][k-1] = state[i+hs-1][k+hs-1][ID_DENS];
            uwnd[i-1][k-1] = state[i+hs-1][k+hs-1][ID_UMOM] / (hy_dens_cell[k] + state[i+hs-1][k+hs-1][ID_DENS]);
            wwnd[i-1][k-1] = state[i+hs-1][k+hs-1][ID_WMOM] / (hy_dens_cell[k] + state[i+hs-1][k+hs-1][ID_DENS]);
            theta[i-1][k-1] = (state[i+hs-1][k+hs-1][ID_RHOT] + hy_dens_theta_cell[k]) / (hy_dens_cell[k] + state[i+hs-1][k+hs-1][ID_DENS]) - hy_dens_theta_cell[k] / hy_dens_cell[k];
        }
    }

    // Write the grid data to file with all the processes writing collectively
    st3[0] = i_beg; st3[1] = k_beg; st3[2] = num_out + 1;
    ct3[0] = nx; ct3[1] = nz; ct3[2] = 1;

    #ifdef SINGLE_PREC
        ncwrap(ncmpi_put_vara_float_all(ncid, dens_varid, st3, ct3, &dens[0][0]), __LINE__);
        ncwrap(ncmpi_put_vara_float_all(ncid, uwnd_varid, st3, ct3, &uwnd[0][0]), __LINE__);
        ncwrap(ncmpi_put_vara_float_all(ncid, wwnd_varid, st3, ct3, &wwnd[0][0]), __LINE__);
        ncwrap(ncmpi_put_vara_float_all(ncid, theta_varid, st3, ct3, &theta[0][0]), __LINE__);
    #else
        ncwrap(ncmpi_put_vara_double_all(ncid, dens_varid, st3, ct3, &dens[0][0]), __LINE__);
        ncwrap(ncmpi_put_vara_double_all(ncid, uwnd_varid, st3, ct3, &uwnd[0][0]), __LINE__);
        ncwrap(ncmpi_put_vara_double_all(ncid, wwnd_varid, st3, ct3, &wwnd[0][0]), __LINE__);
        ncwrap(ncmpi_put_vara_double_all(ncid, theta_varid, st3, ct3, &theta[0][0]), __LINE__);
    #endif

    // Only the main process needs to write the elapsed time
    // Begin "independent" write mode
    ncwrap(ncmpi_begin_indep_data(ncid), __LINE__);

    // Write elapsed time to file
    if (mainproc) {
        st1[0] = num_out + 1; ct1[0] = 1; etimearr[0] = etime;
        #ifdef SINGLE_PREC
            ncwrap(ncmpi_put_vara_float(ncid, t_varid, st1, ct1, etimearr.data()), __LINE__);
        #else
            ncwrap(ncmpi_put_vara_double(ncid, t_varid, st1, ct1, etimearr.data()), __LINE__);
        #endif
    }

    // End "independent" write mode
    ncwrap(ncmpi_end_indep_data(ncid), __LINE__);

    // Close the file
    ncwrap(ncmpi_close(ncid), __LINE__);

    // Increment the number of outputs
    num_out++;
}
