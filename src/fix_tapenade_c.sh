#!/bin/sh
set -eu

usage()
{
  echo "usage:" >&2
  echo "  $0 reverse MODEL_B CONVERT_B" >&2
  echo "  $0 tangent MODEL_D CONVERT_D" >&2
  exit 2
}

fix_reverse()
{
  reverse_model=$1
  reverse_convert=$2

  perl -0pi -e '
    s/#include <adBinomial.h>/#include <adBinomial.h>\n#include <string.h>/;
    s/adBinomial_next\(action, step\)/adBinomial_next(\&action, \&step)/g;

    s/pushReal4\(\*\*\*p\);/pushReal4Array(p[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*p\);/popReal4Array(p[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*vz\);/pushReal4Array(vz[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*vz\);/popReal4Array(vz[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*vx\);/pushReal4Array(vx[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*vx\);/popReal4Array(vx[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*vy\);/pushReal4Array(vy[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*vy\);/popReal4Array(vy[0][0], sim->n123pad);/g;

    s/pushReal4\(\*\*\*xi1\);/pushReal4Array(xi1[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*xi1\);/popReal4Array(xi1[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*xi2\);/pushReal4Array(xi2[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*xi2\);/popReal4Array(xi2[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*xi3\);/pushReal4Array(xi3[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*xi3\);/popReal4Array(xi3[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*memD1p\);/pushReal4Array(memD1p[0][0], 2*sim->nb*sim->n2pad*sim->n3pad);/g;
    s/popReal4\(\*\*memD1p\);/popReal4Array(memD1p[0][0], 2*sim->nb*sim->n2pad*sim->n3pad);/g;
    s/pushReal4\(\*\*\*memD1vz\);/pushReal4Array(memD1vz[0][0], 2*sim->nb*sim->n2pad*sim->n3pad);/g;
    s/popReal4\(\*\*memD1vz\);/popReal4Array(memD1vz[0][0], 2*sim->nb*sim->n2pad*sim->n3pad);/g;
    s/pushReal4\(\*\*\*memD2p\);/pushReal4Array(memD2p[0][0], sim->n1pad*2*sim->nb*sim->n3pad);/g;
    s/popReal4\(\*\*memD2p\);/popReal4Array(memD2p[0][0], sim->n1pad*2*sim->nb*sim->n3pad);/g;
    s/pushReal4\(\*\*\*memD2vx\);/pushReal4Array(memD2vx[0][0], sim->n1pad*2*sim->nb*sim->n3pad);/g;
    s/popReal4\(\*\*memD2vx\);/popReal4Array(memD2vx[0][0], sim->n1pad*2*sim->nb*sim->n3pad);/g;
    s/pushReal4\(\*\*\*memD3p\);/pushReal4Array(memD3p[0][0], sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1);/g;
    s/popReal4\(\*\*memD3p\);/popReal4Array(memD3p[0][0], sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1);/g;
    s/pushReal4\(\*\*\*memD3vy\);/pushReal4Array(memD3vy[0][0], sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1);/g;
    s/popReal4\(\*\*memD3vy\);/popReal4Array(memD3vy[0][0], sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1);/g;
    s/pushReal4\(\*\*\*vpmod\);/pushReal4Array(vpmod[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*vpmod\);/popReal4Array(vpmod[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*rhomod\);/pushReal4Array(rhomod[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*rhomod\);/popReal4Array(rhomod[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*\*qinvmod\);/pushReal4Array(qinvmod[0][0], sim->n123pad);/g;
    s/popReal4\(\*\*qinvmod\);/popReal4Array(qinvmod[0][0], sim->n123pad);/g;
    s/pushReal4\(\*\*dcal\);/pushReal4Array(dcal[0], sim->nt*acq->nrec);/g;
    s/popReal4\(\*dcal\);/popReal4Array(dcal[0], sim->nt*acq->nrec);/g;

    s/\*\*\*pb = 0\.0;/memset(pb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vzb = 0\.0;/memset(vzb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vxb = 0\.0;/memset(vxb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vyb = 0\.0;/memset(vyb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*xi1b = 0\.0;/memset(xi1b[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*xi2b = 0\.0;/memset(xi2b[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*xi3b = 0\.0;/memset(xi3b[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*kappab = 0\.0;/memset(kappab[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*buzb = 0\.0;/memset(buzb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*buxb = 0\.0;/memset(buxb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*buyb = 0\.0;/memset(buyb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*qinvmodb = 0\.0;/memset(qinvmodb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*memD1pb = 0\.0;/memset(memD1pb[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD1vzb = 0\.0;/memset(memD1vzb[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD2pb = 0\.0;/memset(memD2pb[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD2vxb = 0\.0;/memset(memD2vxb[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD3pb = 0\.0;/memset(memD3pb[0][0], 0, (sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1)*sizeof(float));/g;
    s/\*\*\*memD3vyb = 0\.0;/memset(memD3vyb[0][0], 0, (sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1)*sizeof(float));/g;
    s/\*\*\*qinvb = 0\.0;/memset(qinvb[0][0], 0, sim->n123*sizeof(float));/g;
    s/\*\*\*vpb = 0\.0;/memset(vpb[0][0], 0, sim->n123*sizeof(float));/g;
    s/\*\*\*rhob = 0\.0;/memset(rhob[0][0], 0, sim->n123*sizeof(float));/g;
    s/\*\*dcalb = 0\.0;/memset(dcalb[0], 0, sim->nt*acq->nrec*sizeof(float));/g;

    # Tapenade emits this reverse replay loop without braces. Keep the
    # generated control flow explicit and leave label110 in the time loop.
    s/if \(action != 4\)\n            for \(i = 1; i < adCount\+1; \+\+i\)\n                if \(i == 1\) \{/if (action != 4) {\n            for (i = 1; i < adCount+1; ++i) {\n                if (i == 1) {/;
    s/\n  label110: ;/\n            }\n        }\n  label110: ;/;
  ' "$reverse_model"

  if grep -Eq 'adBinomial_next\(action|pushReal4\(\*\*|popReal4\(\*|\*\*\*[^;]+ = 0\.0|\*\*dcalb = 0\.0' "$reverse_model"; then
    echo "unrepaired Tapenade pointer-array operation in $reverse_model" >&2
    exit 1
  fi

  perl -0pi -e '
    s/#include <adStack.h>/#include <adStack.h>\n#include <string.h>/;
    s/\*\*\*vpmodb = 0\.0;/memset(vpmodb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*rhomodb = 0\.0;/memset(rhomodb[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vpb = 0\.0;/memset(vpb[0][0], 0, sim->n123*sizeof(float));/g;
    s/\*\*\*rhob = 0\.0;/memset(rhob[0][0], 0, sim->n123*sizeof(float));/g;
    s/\*\*\*qinvb = 0\.0;/memset(qinvb[0][0], 0, sim->n123*sizeof(float));/g;
    s/^    float tmp(?:1|4|7|10|13|16);\n//mg;
  ' "$reverse_convert"

  if grep -Eq '\*\*\*(vpmodb|rhomodb|vpb|rhob) = 0\.0' "$reverse_convert"; then
    echo "unrepaired Tapenade pointer-array operation in $reverse_convert" >&2
    exit 1
  fi
}

fix_tangent()
{
  tangent_model=$1
  tangent_convert=$2

  perl -0pi -e '
    s/#include "sim.h"/#include "sim.h"\n#include <string.h>/ if $. == 1;
    s/\*\*\*qinvmodd = 0\.0;/memset(qinvmodd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vpmodd = 0\.0;/memset(vpmodd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*rhomodd = 0\.0;/memset(rhomodd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*kappad = 0\.0;/memset(kappad[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*buzd = 0\.0;/memset(buzd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*buxd = 0\.0;/memset(buxd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*buyd = 0\.0;/memset(buyd[0][0], 0, sim->n123pad*sizeof(float));/g;
  ' "$tangent_convert"

  perl -0pi -e '
    s/#include "sim.h"/#include "sim.h"\n#include <string.h>/ if $. == 1;
    s/\*\*\*pd = 0\.0;/memset(pd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vzd = 0\.0;/memset(vzd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vxd = 0\.0;/memset(vxd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*vyd = 0\.0;/memset(vyd[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*memD1pd = 0\.0;/memset(memD1pd[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD2pd = 0\.0;/memset(memD2pd[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD3pd = 0\.0;/memset(memD3pd[0][0], 0, (sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1)*sizeof(float));/g;
    s/\*\*\*memD1vzd = 0\.0;/memset(memD1vzd[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD2vxd = 0\.0;/memset(memD2vxd[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));/g;
    s/\*\*\*memD3vyd = 0\.0;/memset(memD3vyd[0][0], 0, (sim->n3>1 ? sim->n1pad*sim->n2pad*2*sim->nb : 1)*sizeof(float));/g;
    s/\*\*\*xi1d = 0\.0;/memset(xi1d[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*xi2d = 0\.0;/memset(xi2d[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*\*xi3d = 0\.0;/memset(xi3d[0][0], 0, sim->n123pad*sizeof(float));/g;
    s/\*\*dcald = 0\.0;/memset(dcald[0], 0, sim->nt*acq->nrec*sizeof(float));/g;
  ' "$tangent_model"

  if grep -Eq '\*\*\*(pd|vzd|vxd|vyd|memD1pd|memD2pd|memD3pd|memD1vzd|memD2vxd|memD3vyd|xi1d|xi2d|xi3d|vpmodd|rhomodd|qinvmodd|kappad|buzd|buxd|buyd) = 0\.0|\*\*dcald = 0\.0' \
      "$tangent_model" "$tangent_convert"; then
    echo "unrepaired Tapenade tangent pointer-array operation" >&2
    exit 1
  fi
}

case ${1-} in
  reverse)
    [ "$#" -eq 3 ] || usage
    fix_reverse "$2" "$3"
    ;;
  tangent)
    [ "$#" -eq 3 ] || usage
    fix_tangent "$2" "$3"
    ;;
  *)
    usage
    ;;
esac
