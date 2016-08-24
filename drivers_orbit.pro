pro drivers_orbit

READ, orbit_0, PROMPT='Enter Orbit Number 1: '
READ, orbit_1, PROMPT='Enter Orbit Number 2: '
trange = [time_string(mvn_orbit_num(orbnum=orbit_0-1)), time_string(mvn_orbit_num(orbnum=orbit_1+1))]
          
tlimit, trange
end