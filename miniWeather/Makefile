# Main Makefile

# Default target
all:
	@echo "ERROR: no target is provided."
	@echo "Usage: make d<num> | c<num>"
	@exit -1

d%:
	@$(MAKE) -C $*/fortran do
	@$(MAKE) -C $*/cpp do

c%:
	@$(MAKE) -C $*/fortran clean 
	@$(MAKE) -C $*/cpp clean 

f%:
	@$(MAKE) -C $*/fortran clean do
	@$(MAKE) -C $*/cpp clean  do

.PHONY: all
