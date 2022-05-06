HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=/opt/rocm
endif
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++14 -O0 -I/opt/include/

g1_add: g1_add.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

.PHONY: clean

clean:
	rm -f g1_add *.o