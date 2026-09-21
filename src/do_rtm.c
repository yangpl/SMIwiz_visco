/* Reverse-time migration through the existing FWI impedance-gradient kernel. */
#include "sim.h"
#include "acq.h"

void do_fwi(sim_t *sim, acq_t *acq);

void do_rtm(sim_t *sim, acq_t *acq)
/*< form the family-2 logarithmic impedance migration image >*/
{
  do_fwi(sim, acq);
}
