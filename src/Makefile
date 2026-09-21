CC = mpicc
CFLAGS = -O2 -Wall
TAPENADE_HOME ?= /home/pyang/Install/tapenade_3.16
TAPENADE = $(TAPENADE_HOME)/bin/tapenade
ADKIT = $(TAPENADE_HOME)/ADFirstAidKit
AD_OUT = ./tapenade_out
AD_SRC = $(AD_OUT)/modelling_b.c $(AD_OUT)/extend_model_b.c \
	$(AD_OUT)/fdtd_b.c $(AD_OUT)/inject_extract_b.c
AD_OBJ = $(AD_SRC:.c=.o)
AD_STAMP = $(AD_OUT)/.adjoint_generated
AD_HEAD = modelling(dcal)/(vp rho qinv)
AD_PRIMAL_SRC = modelling.c extend_model.c fdtd.c inject_extract.c
AD_PRIMAL_DIR = $(AD_OUT)/ad_primal
AD_TAPENADE_SRC = $(AD_PRIMAL_SRC:%=$(AD_PRIMAL_DIR)/%)
AD_FIXER = fix_tapenade_c.sh

TLM_SRC = $(AD_OUT)/modelling_d.c $(AD_OUT)/extend_model_d.c \
	$(AD_OUT)/fdtd_d.c $(AD_OUT)/inject_extract_d.c
TLM_OBJ = $(TLM_SRC:.c=.o)
TLM_STAMP = $(AD_OUT)/.tangent_generated
TLM_HEAD = modelling(dcal)/(vp rho qinv)

fftw3 ?= /usr
fftw3_lib = $(fftw3)/lib
fftw3_inc = $(fftw3)/include

BIN = ../bin
LIB = -L$(fftw3_lib) -lfftw3 -lm -fopenmp -lpthread  -lmpi
INC = -I../include -I$(fftw3_inc)
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)
AD_RUNTIME_OBJ = $(AD_OUT)/adStack.o $(AD_OUT)/adBinomial.o
COMMON_HEADERS = ../include/acq.h ../include/fwi.h ../include/modelling.h \
	../include/opt.h ../include/sim.h



.PHONY: all SMIwiz adjoint-generate tlm-generate tangent-generate clean

all:
	$(MAKE) clean
	$(MAKE) SMIwiz

%.o: %.c $(COMMON_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) $(LIB)

SMIwiz: $(BIN)/SMIwiz

$(BIN)/SMIwiz: $(OBJ) $(AD_OBJ) $(TLM_OBJ) $(AD_RUNTIME_OBJ)
	mkdir -p $(BIN)
	$(CC) $(CFLAGS) -o $(BIN)/SMIwiz $(OBJ) $(AD_OBJ) $(TLM_OBJ) $(AD_RUNTIME_OBJ) $(LIB)

adjoint-generate: $(AD_SRC) $(AD_STAMP)

tlm-generate: $(TLM_SRC) $(TLM_STAMP)

tangent-generate: tlm-generate

$(AD_PRIMAL_DIR)/%.c: %.c Makefile
	mkdir -p $(AD_PRIMAL_DIR)
	awk 'BEGIN{skip=0} /TAPENADE-IGNORE-BEGIN/{skip=1; next} /TAPENADE-IGNORE-END/{skip=0; next} !skip{print}' $< > $@

$(AD_SRC) $(AD_STAMP) &: $(AD_TAPENADE_SRC) \
	../include/sim.h ../include/acq.h \
	$(AD_FIXER) Makefile
	mkdir -p $(AD_OUT)
	$(RM) $(AD_SRC) $(AD_OUT)/modelling_b.msg $(AD_OUT)/sim_b.h $(AD_OUT)/*~
	env _JAVA_OPTIONS=-XX:+PerfDisableSharedMem $(TAPENADE) -reverse \
		-head '$(AD_HEAD)' -outputdirectory $(AD_OUT) -I../include \
		$(AD_TAPENADE_SRC)
	sh $(AD_FIXER) reverse $(AD_OUT)/modelling_b.c \
		$(AD_OUT)/extend_model_b.c
	touch $(AD_STAMP)

$(TLM_SRC) $(TLM_STAMP) &: $(AD_TAPENADE_SRC) \
	../include/sim.h ../include/acq.h ../include/modelling.h \
	$(AD_FIXER) Makefile $(AD_STAMP)
	mkdir -p $(AD_OUT)
	$(RM) $(TLM_SRC) $(AD_OUT)/modelling_d.msg $(AD_OUT)/sim_d.h $(AD_OUT)/*~
	env _JAVA_OPTIONS=-XX:+PerfDisableSharedMem $(TAPENADE) -d \
		-head '$(TLM_HEAD)' -outputdirectory $(AD_OUT) -I../include \
		$(AD_TAPENADE_SRC)
	sh $(AD_FIXER) tangent $(AD_OUT)/modelling_d.c \
		$(AD_OUT)/extend_model_d.c
	touch $(TLM_STAMP)

$(AD_OUT)/%.o: $(AD_OUT)/%.c $(AD_STAMP) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) -I$(ADKIT)

$(AD_OUT)/adStack.o: $(ADKIT)/adStack.c
	mkdir -p $(AD_OUT)
	$(CC) $(CFLAGS) -c $< -o $@ -I$(ADKIT)

$(AD_OUT)/adBinomial.o: $(ADKIT)/adBinomial.c
	mkdir -p $(AD_OUT)
	$(CC) $(CFLAGS) -c $< -o $@ -I$(ADKIT)

$(TLM_OBJ): $(AD_OUT)/%.o: $(AD_OUT)/%.c $(TLM_STAMP) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) -I$(ADKIT)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(OBJ) $(BIN)/SMIwiz
	rm -rf $(AD_OUT)
