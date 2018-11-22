/***************************************************/
/* float vector and matrix operations
/*	Copyright(c) Toshihiro Matsui
/*      Electrotechnical Laboratory
/*
/*	1986-Nov
/*	1987-Feb	complete floatvector
/*	1987-Mar	modify rotation 
/*	1987-Nov	simultaneous equations
/*	1988-Jan	matrix is represented by 2D array,
/*			not by an vector of vectors
/**************************************************************/
static char *rcsid="@(#)$Id$";

#include "math.h"
#include "eus.h"

extern pointer makematrix();

#define ismatrix(p) ((isarray(p) && \
                      p->c.ary.rank==makeint(2) && \
                      elmtypeof(p->c.ary.entity)==ELM_FLOAT))
#define rowsize(p) (intval(p->c.ary.dim[0]))
#define colsize(p) (intval(p->c.ary.dim[1]))

static pointer K_X,K_Y,K_Z,MK_X,MK_Y,MK_Z;

#pragma init (register_matprod)

extern pointer ___matprod();
static register_matprod()
{ add_module_initializer("___matprod", ___matprod);}

pointer MATPROD(ctx,n,argv)
register context *ctx;
int n;
register pointer argv[];
{ pointer rm;
  register int i,j,ii,row1,column1,row2,column2;
  register eusfloat_t *fm1,*fm2,*fm;
  eusfloat_t *fv,x,fvv[256];
  fv = fvv;

  ckarg2(2,3);
  if (!ismatrix(argv[0]) || !ismatrix(argv[1])) error(E_NOVECTOR);
  fm1=argv[0]->c.ary.entity->c.fvec.fv;
  fm2=argv[1]->c.ary.entity->c.fvec.fv;
  row1=rowsize(argv[0]);	row2=rowsize(argv[1]);
  column1=colsize(argv[0]); 	column2=colsize(argv[1]);
  if (column1!=column2 || row1!=row2) error(E_VECINDEX);
  if (n==3) {
    rm=argv[2];
    if (!ismatrix(rm)) error(E_NOVECTOR);
    if (row1!=rowsize(rm) || column1!=colsize(rm)) error(E_VECINDEX);
    }
  else rm=makematrix(ctx,row1,column1);
  fm=rm->c.ary.entity->c.fvec.fv;
  for (i=0; i<row1; i++) {
    for (j=0; j<column1; j++) {
      ii=i*column1+j;
      fm[ii]=fm1[ii]*fm2[ii];
    }
  }
  return(rm);}


pointer ___matprod(register context *ctx, int n, register pointer *argv)
{
  pointer mod=argv[0];
  defun(ctx,"MPROD",mod,MATPROD,NULL);}
