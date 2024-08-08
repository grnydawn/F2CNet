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
        state_tmp = state;
        flux.resize(nx + 1, std::vector<std::vector<real>>(nz + 1, std::vector<real>(NUM_VARS)));
        tend.resize(nx, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));

        sendbuf_l.resize(hs, std::vector<std::vector<real>>(nz, std::vector<real>(NUM_VARS)));
        sendbuf_r = sendbuf_l;
        recvbuf_l = sendbuf_l;
        recvbuf_r = sendbuf_l;

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
}

int main(int argc, char** argv) {
    // Initialize the MiniWeather simulation
    MiniWeather simulation;
    simulation.init();

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
