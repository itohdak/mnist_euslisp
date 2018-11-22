include Makefile.$(ARCHDIR)

matprodDIR		= src

CFLAGS		= -I. -I.. -fPIC -DLinux -w -falign-functions=8 -DGCC3 -DGCC -DTHREADED -DPTHREAD -fpic -D$(ARCHDIR) -I$(EUSDIR)/include
LDFLAGS		= -L. -L..

.SUFFIXES: .c

SRCS		= matprod.c

OBJECTS		= $(ARCHDIR)/matprod.o

CLEAN		= $(ARCHDIR)/$(LPFX)matprod.$(LSFX) $(OBJECTS) *.$(LSFX) \
		$(ARCHDIR)/*.o *~ $(ARCHDIR)/$(LPFX)matprod-static.a

default: directory library

directory:
	if [ ! -e $(ARCHDIR) ]; then mkdir -p $(ARCHDIR); fi

install: $(EUSDIR)/$(ARCHDIR)/lib/$(LPFX)matprod.$(LSFX)

$(EUSDIR)/$(ARCHDIR)/lib/$(LPFX)matprod.$(LSFX): $(ARCHDIR)/$(LPFX)matprod.$(LSFX)
	cp $(ARCHDIR)/$(LPFX)matprod.$(LSFX) $(EUSDIR)/$(ARCHDIR)/lib

library: $(ARCHDIR)/$(LPFX)matprod.$(LSFX) \
	$(ARCHDIR)/$(LPFX)matprod-static.a

$(ARCHDIR)/$(LPFX)matprod.$(LSFX): $(OBJECTS)
	$(LD) $(SOFLAGS) $(OUTOPT)$(ARCHDIR)/$(LPFX)matprod.$(LSFX) $(OBJECTS) $(EUSDIR)/$(ARCHDIR)/lib/libeusx.so

$(ARCHDIR)/$(LPFX)matprod-static.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $(OBJECTS) $(EUSDIR)/$(ARCHDIR)/obj/eus.o

$(OBJECTS):
	$(CC) $(CFLAGS) -DCOMPILE_LIB -c $(matprodDIR)/$(*F).c $(OBJOPT)$@

clean:
	rm -f $(CLEAN)
