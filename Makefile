# GNU Make solution makefile autogenerated by Premake
# Type "make help" for usage help

ifndef config
  config=release
endif
export config

PROJECTS := SignedDistFunc

.PHONY: all clean help $(PROJECTS)

all: $(PROJECTS)

SignedDistFunc: 
	@echo "==== Building SignedDistFunc ($(config)) ===="
	@${MAKE} --no-print-directory -C . -f SignedDistFunc.make

clean:
	@${MAKE} --no-print-directory -C . -f SignedDistFunc.make clean

help:
	@echo "Usage: make [config=name] [target]"
	@echo ""
	@echo "CONFIGURATIONS:"
	@echo "   release"
	@echo "   debug"
	@echo ""
	@echo "TARGETS:"
	@echo "   all (default)"
	@echo "   clean"
	@echo "   SignedDistFunc"
	@echo ""
	@echo "For more information, see http://industriousone.com/premake/quick-start"
