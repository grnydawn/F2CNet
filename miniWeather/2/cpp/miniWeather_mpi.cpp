#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <pnetcdf.h>

// Precision for real numbers
#ifdef SINGLE_PREC
using real = float;
const MPI_Datatype MPI_REAL_TYPE = MPI_FLOAT;
#else
using real = double;
const MPI_Datatype MPI_REAL_TYPE = MPI_DOUBLE;
#endif

// Physical constants
constexpr real pi        = 3.14159265358979323846;
constexpr real grav      = 9.8;
constexpr real cp        = 1004.0;
constexpr real cv        = 717.0;
constexpr real rd        = 287.0;
constexpr real p0        = 1.0e5;
constexpr real C0        = 27.5629410929725921310572974482;
constexpr real gamma     = 1.40027894002789400278940027894;

// Domain and stability-related constants
constexpr real xlen      = 2.0e4;
constexpr real zlen      = 1.0e4;
constexpr real hv_beta   = 0.05;
constexpr real cfl       = 1.50;

constexpr real max_speed = 450;
constexpr int  hs        = 2;
constexpr int  sten_size = 4;

// Indexing and flags
constexpr int NUM_VARS = 4;
constexpr int ID_DENS  = 0;
constexpr int ID_UMOM  = 1;
constexpr int ID_WMOM  = 2;
constexpr int ID_RHOT  = 3;
constexpr int DIR_X = 1;
constexpr int DIR_Z = 2;
constexpr int DATA_SPEC_COLLISION       = 1;
constexpr int DATA_SPEC_THERMAL         = 2;
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3;
constexpr int DATA_SPEC_DENSITY_CURRENT = 5;
constexpr int DATA_SPEC_INJECTION       = 6;

// Gauss-Legendre quadrature points and weights
constexpr int nqpoints = 3;
const real qpoints[nqpoints]  = {0.112701665379258311482073460022, 0.500000000000000000000000000000, 0.887298334620741688517926539980};
const real qweights[nqpoints] = {0.277777777777777777777777777779, 0.444444444444444444444444444444, 0.277777777777777777777777777779};

// User-configurable parameters
constexpr int nx_glob = _NX;
constexpr int nz_glob = _NZ;
constexpr real sim_time = _SIM_TIME;
constexpr real output_freq = _OUT_FREQ;
constexpr int data_spec_int = _DATA_SPEC;

void hydro_const_theta(real z, real &r, real &t) {
    const real theta0 = 300.0;
    const real exner0 = 1.0;
    real p, exner, rt;

    t = theta0;
    exner = exner0 - grav * z / (cp * theta0);
    p = p0 * std::pow(exner, cp/rd);
    rt = std::pow(p / C0, 1.0 / gamma);
    r = rt / t;
}

void hydro_const_bvfreq(real z, real bv_freq0, real &r, real &t) {
    real theta0 = 300, exner0 = 1;
    real p, exner, rt;

    t = theta0 * std::exp(bv_freq0 * bv_freq0 / grav * z);
    exner = exner0 - grav * grav / (cp * bv_freq0 * bv_freq0) * (t - theta0) / (t * theta0);
    p = p0 * std::pow(exner, cp/rd);
    rt = std::pow(p / C0, 1.0 / gamma);
    r = rt / t;
}

real sample_ellipse_cosine(real x, real z, real amp, real x0, real z0, real xrad, real zrad) {
    real dist = std::sqrt(std::pow((x-x0)/xrad, 2) + std::pow((z-z0)/zrad, 2)) * pi / 2.0;
    if (dist <= pi / 2.0) {
        return amp * std::pow(std::cos(dist), 2);
    } else {
        return 0.0;
    }
}

//Meta and Claude
void injection(real x, real z, real& r, real& u, real& w, real& t, real& hr, real& ht) {
    hydro_const_theta(z, hr, ht);
    r = 0;
    t = 0;
    u = 0;
    w = 0;
}

//Meta and Claude
void density_current(real x, real z, real& r, real& u, real& w, real& t, real& hr, real& ht) {
    hydro_const_theta(z, hr, ht);
    r = 0;
    t = 0;
    u = 0;
    w = 0;
    t += sample_ellipse_cosine(x, z, -20.0, xlen / 2, 5000.0, 4000.0, 2000.0);
}

void gravity_waves(real x, real z, real& r, real& u, real& w, real& t, real& hr, real& ht) {
    hydro_const_bvfreq(z, 0.02, hr, ht);
    r = 0;
    t = 0;
    u = 15;
    w = 0;
}

void thermal(real x, real z, real& r, real& u, real& w, real& t, real& hr, real& ht) {
    hydro_const_theta(z, hr, ht);
    r = 0;
    t = 0;
    u = 0;
    w = 0;
    t += sample_ellipse_cosine(x, z, 3.0, xlen / 2, 2000.0, 2000.0, 2000.0);
}

void collision(real x, real z, real& r, real& u, real& w, real& t, real& hr, real& ht) {
    hydro_const_theta(z, hr, ht);
    r = 0;
    t = 0;
    u = 0;
    w = 0;
    t += sample_ellipse_cosine(x, z, 20.0, xlen / 2, 2000.0, 2000.0, 2000.0);
    t += sample_ellipse_cosine(x, z, -20.0, xlen / 2, 8000.0, 2000.0, 2000.0);
}

class MiniWeather {
public:
    MiniWeather() {
        init();
    }

    ~MiniWeather() {
        finalize();
    }

private:
    // Member variables (formerly global variables)
    real dt, dx, dz;
    int nx, nz, i_beg, k_beg;
    int nranks, myrank, left_rank, right_rank;
    bool mainproc;
    std::vector<real> hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int;
    real etime, output_counter;
    //CHECK: three dimensional vectors???
    std::vector<std::vector<std::vector<real>>> state, state_tmp, flux, tend;
    std::vector<std::vector<std::vector<real>>> sendbuf_l, sendbuf_r, recvbuf_l, recvbuf_r;

    void init() {
        // Initialize MPI
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // Set up domain decomposition
        real nper = static_cast<real>(nx_glob) / nranks;
        i_beg = std::round(nper * myrank) + 1;
        int i_end = std::round(nper * (myrank + 1));
        nx = i_end - i_beg + 1;
        left_rank = (myrank - 1 + nranks) % nranks;
        right_rank = (myrank + 1) % nranks;

        // Set up vertical domain
        k_beg = 1;
        nz = nz_glob;
        mainproc = (myrank == 0);

        // Allocate arrays
        dx = xlen / nx_glob;
        dz = zlen / nz_glob;
        hy_dens_cell.resize(nz + 2*hs);
        hy_dens_theta_cell.resize(nz + 2*hs);
        hy_dens_int.resize(nz + 1);
        hy_dens_theta_int.resize(nz + 1);
        hy_pressure_int.resize(nz + 1);

        state.resize(nx + 2*hs, std::vector<std::vector<real>>(nz + 2*hs, std::vector<real>(NUM_VARS)));
        state_tmp.resize(nx + 2*hs, std::vector<std::vector<real>>(nz + 2*hs, std::vector<real>(NUM_VARS)));
        flux.resize(nx + 1, std::vector<std::vector<real>>(nz + 1, std::vector<real>(NUM_VARS)));
        tend.resize(nx, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));

        sendbuf_l.resize(hs, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));
        sendbuf_r.resize(hs, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));
        recvbuf_l.resize(hs, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));
        recvbuf_r.resize(hs, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));

        // Initialize dt, etime, and output_counter
        dt = std::min(dx, dz) / max_speed * cfl;
        etime = 0;
        output_counter = 0;

        if (mainproc) {
            std::cout << "nx_glob, nz_glob: " << nx_glob << ", " << nz_glob << std::endl;
            std::cout << "dx, dz: " << dx << ", " << dz << std::endl;
            std::cout << "dt: " << dt << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Initialize the fluid state
        for (int k = 1-hs; k <= nz+hs; k++) {
            for (int i = 1-hs; i <= nx+hs; i++) {
                for (int kk = 0; kk < nqpoints; kk++) {
                    for (int ii = 0; ii < nqpoints; ii++) {
                        real x = (i_beg-1 + i-0.5) * dx + (qpoints[ii]-0.5)*dx;
                        real z = (k_beg-1 + k-0.5) * dz + (qpoints[kk]-0.5)*dz;

                        real r, u, w, t, hr, ht;
                        if (data_spec_int == DATA_SPEC_COLLISION) {
                            collision(x, z, r, u, w, t, hr, ht);
                        } else if (data_spec_int == DATA_SPEC_THERMAL) {
                            thermal(x, z, r, u, w, t, hr, ht);
                        } else if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
                            gravity_waves(x, z, r, u, w, t, hr, ht);
                        } else if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) {
                            density_current(x, z, r, u, w, t, hr, ht);
                        } else if (data_spec_int == DATA_SPEC_INJECTION) {
                            injection(x, z, r, u, w, t, hr, ht);
                        }

                        state[i+hs-1][k+hs-1][ID_DENS] += r * qweights[ii] * qweights[kk];
                        state[i+hs-1][k+hs-1][ID_UMOM] += (r+hr) * u * qweights[ii] * qweights[kk];
                        state[i+hs-1][k+hs-1][ID_WMOM] += (r+hr) * w * qweights[ii] * qweights[kk];
                        state[i+hs-1][k+hs-1][ID_RHOT] += ((r+hr)*(t+ht) - hr*ht) * qweights[ii] * qweights[kk];
                    }
                }
                for (int ll = 0; ll < NUM_VARS; ll++) {
                    state_tmp[i+hs-1][k+hs-1][ll] = state[i+hs-1][k+hs-1][ll];
                }
            }
        }

        // Compute hydrostatic background state
        for (int k = 1-hs; k <= nz+hs; k++) {
            for (int kk = 0; kk < nqpoints; kk++) {
                real z = (k_beg-1 + k-0.5) * dz + (qpoints[kk]-0.5)*dz;
                real r, u, w, t, hr, ht;
                if (data_spec_int == DATA_SPEC_COLLISION) {
                    collision(0, z, r, u, w, t, hr, ht);
                } else if (data_spec_int == DATA_SPEC_THERMAL) {
                    thermal(0, z, r, u, w, t, hr, ht);
                } else if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
                    gravity_waves(0, z, r, u, w, t, hr, ht);
                } else if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) {
                    density_current(0, z, r, u, w, t, hr, ht);
                } else if (data_spec_int == DATA_SPEC_INJECTION) {
                    injection(0, z, r, u, w, t, hr, ht);
                }
                hy_dens_cell[k+hs-1] += hr * qweights[kk];
                hy_dens_theta_cell[k+hs-1] += hr*ht * qweights[kk];
            }
        }

        for (int k = 1; k <= nz+1; k++) {
            real z = (k_beg-1 + k-1) * dz;
            real r, u, w, t, hr, ht;
            if (data_spec_int == DATA_SPEC_COLLISION) {
                collision(0, z, r, u, w, t, hr, ht);
            } else if (data_spec_int == DATA_SPEC_THERMAL) {
                thermal(0, z, r, u, w, t, hr, ht);
            } else if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
                gravity_waves(0, z, r, u, w, t, hr, ht);
            } else if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) {
                density_current(0, z, r, u, w, t, hr, ht);
            } else if (data_spec_int == DATA_SPEC_INJECTION) {
                injection(0, z, r, u, w, t, hr, ht);
            }
            hy_dens_int[k-1] = hr;
            hy_dens_theta_int[k-1] = hr*ht;
            hy_pressure_int[k-1] = C0 * std::pow(hr*ht, gamma);
        }
    }

    void finalize() {
        // Assuming these are now std::vector or other C++ container types
        state.clear();
        state_tmp.clear();
        flux.clear();
        tend.clear();
        hy_dens_cell.clear();
        hy_dens_theta_cell.clear();
        hy_dens_int.clear();
        hy_dens_theta_int.clear();
        hy_pressure_int.clear();
        sendbuf_l.clear();
        sendbuf_r.clear();
        recvbuf_l.clear();
        recvbuf_r.clear();
        
        MPI_Finalize();
    }
};

// Claude only
void ncwrap(int ierr, int line) {
    if (ierr != NC_NOERR) {
        std::cout << "NetCDF Error at line: " << line << std::endl;
        std::cout << ncmpi_strerror(ierr) << std::endl;
        exit(1);
    }
}

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
void reductions(real &mass, real &te) {
    mass = 0;
    te = 0;

    #pragma acc parallel loop collapse(2) reduction(+:mass,te)
    for (int k = 1; k <= nz; k++) {
        for (int i = 1; i <= nx; i++) {
            real r = state[i][k][ID_DENS] + hy_dens_cell[k];
            real u = state[i][k][ID_UMOM] / r;
            real w = state[i][k][ID_WMOM] / r;
            real th = (state[i][k][ID_RHOT] + hy_dens_theta_cell[k]) / r;
            real p = C0 * std::pow(r*th, gamma);
            real t = th / std::pow(p0/p, rd/cp);
            real ke = r * (u*u + w*w);
            real ie = r * cv * t;
            mass += r * dx * dz;
            te += (ke + r*cv*t) * dx * dz;
        }
    }

    std::array<real, 2> glob;
    MPI_Allreduce({mass, te}, glob.data(), 2, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD);
    mass = glob[0];
    te = glob[1];
}

int main(int argc, char** argv) {
    // Initialize the MiniWeather simulation
    MiniWeather simulation;
    //simulation.init();

    // Initial reductions for mass, kinetic energy, and total energy
    real mass0, te0;
    simulation.reductions(mass0, te0);

    // Output the initial state
    simulation.output(simulation.state, simulation.etime);

    // Main time step loop
    int64_t t1, t2;
    if (simulation.mainproc) t1 = MPI_Wtime();
    while (simulation.etime < simulation.sim_time) {
        // If the time step leads to exceeding the simulation time, shorten it for the last step
        if (simulation.etime + simulation.dt > simulation.sim_time) simulation.dt = simulation.sim_time - simulation.etime;

        // Perform a single time step
        simulation.perform_timestep();

        // Inform the user
#ifndef NO_INFORM
        if (simulation.mainproc) std::cout << "Elapsed Time: " << simulation.etime << " / " << simulation.sim_time << std::endl;
#endif

        // Update the elapsed time and output counter
        simulation.etime += simulation.dt;
        simulation.output_counter += simulation.dt;

        // If it's time for output, reset the counter, and do output
        if (simulation.output_counter >= simulation.output_freq) {
            simulation.output_counter -= simulation.output_freq;
            simulation.output(simulation.state, simulation.etime);
        }
    }

    if (simulation.mainproc) {
        t2 = MPI_Wtime();
        std::cout << "CPU Time: " << (t2 - t1) << std::endl;
    }

    // Final reductions for mass, kinetic energy, and total energy
    real mass, te;
    simulation.reductions(mass, te);

    if (simulation.mainproc) {
        std::cout << "d_mass: " << (mass - mass0) / mass0 << std::endl;
        std::cout << "d_te:   " << (te - te0) / te0 << std::endl;
    }

    // Deallocate and finalize MPI
    simulation.finalize();

    return 0;
}
