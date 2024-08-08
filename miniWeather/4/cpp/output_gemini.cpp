#include <iostream>
#include <netcdf>
#include <vector>

// ... (other includes and type definitions as needed)

// Output function
void output(const std::vector<std::vector<std::vector<double>>>& state, double etime) {
    using namespace netCDF;
    using namespace netCDF::exceptions; 
    
    if (mainproc) std::cout << "*** OUTPUT ***" << std::endl;

    static int num_out = 0; 
    int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid;
    std::vector<size_t> start, count;
    std::vector<double> etime_arr(1);
    etime_arr[0] = etime;

    // Create temporary arrays for storing data
    std::vector<std::vector<double>> dens(nx, std::vector<double>(nz));
    std::vector<std::vector<double>> uwnd(nx, std::vector<double>(nz));
    std::vector<std::vector<double>> wwnd(nx, std::vector<double>(nz));
    std::vector<std::vector<double>> theta(nx, std::vector<double>(nz));

    try {
        // Open or create NetCDF file
        if (etime == 0) {
            NcFile file("output.nc", NcFile::replace);
            t_dimid = file.addDim("t", NC_UNLIMITED);
            x_dimid = file.addDim("x", nx_glob);
            z_dimid = file.addDim("z", nz_glob);
            dens_varid = file.addVar("dens", NC_DOUBLE, {x_dimid, z_dimid, t_dimid});
            uwnd_varid = file.addVar("uwnd", NC_DOUBLE, {x_dimid, z_dimid, t_dimid});
            wwnd_varid = file.addVar("wwnd", NC_DOUBLE, {x_dimid, z_dimid, t_dimid});
            theta_varid = file.addVar("theta", NC_DOUBLE, {x_dimid, z_dimid, t_dimid});
            t_varid = file.addVar("t", NC_DOUBLE, {t_dimid});
        } else {
            NcFile file("output.nc", NcFile::write);
            dens_varid = file.getVar("dens").getId();
            uwnd_varid = file.getVar("uwnd").getId();
            wwnd_varid = file.getVar("wwnd").getId();
            theta_varid = file.getVar("theta").getId();
            t_varid = file.getVar("t").getId();
        }

        // Store data in temporary arrays
        for (int k = 0; k < nz; k++) {
            for (int i = 0; i < nx; i++) {
                dens[i][k] = state[i][k][ID_DENS];
                uwnd[i][k] = state[i][k][ID_UMOM] / (hy_dens_cell[k] + state[i][k][ID_DENS]);
                wwnd[i][k] = state[i][k][ID_WMOM] / (hy_dens_cell[k] + state[i][k][ID_DENS]);
                theta[i][k] = (state[i][k][ID_RHOT] + hy_dens_theta_cell[k]) / (hy_dens_cell[k] + state[i][k][ID_DENS]) - hy_dens_theta_cell[k] / hy_dens_cell[k];
            }
        }

        // Write data to NetCDF file
        start = {i_beg, k_beg, num_out}; 
        count = {nx, nz, 1};

        file.putVar(dens_varid, start, count, &dens[0][0]);
        file.putVar(uwnd_varid, start, count, &uwnd[0][0]);
        file.putVar(wwnd_varid, start, count, &wwnd[0][0]);
        file.putVar(theta_varid, start, count, &theta[0][0]);

        if (mainproc) {
            start = {num_out}; count = {1};
            file.putVar(t_varid, start, count, &etime_arr[0]);
        }

    } catch (NcException& e) {
        std::cerr << "NetCDF error: " << e.what() << std::endl;
    }

    num_out++; // Increment output counter
}

