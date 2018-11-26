#include "math.h"
#include "eus.h"

extern pointer makematrix();

#define ismatrix(p) ((isarray(p) && \
                      p->c.ary.rank==makeint(2) && \
                      elmtypeof(p->c.ary.entity)==ELM_FLOAT))
#define rowsize(p) (intval(p->c.ary.dim[0]))
#define colsize(p) (intval(p->c.ary.dim[1]))

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
  eusfloat_t *fv,x,fvv[10000];
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

pointer MATRELU(ctx,n,argv)
     register context *ctx;
     int n;
     register pointer argv[];
{ pointer rm;
  register int i,j,ii,row1,column1;
  register eusfloat_t *fm1,*fm;
  int flag;
  eusfloat_t *fv,x,fvv[10000];
  fv = fvv;

  ckarg2(2,3);
  flag = ckintval(argv[0]);
  if (!ismatrix(argv[1])) error(E_NOVECTOR);
  fm1=argv[1]->c.ary.entity->c.fvec.fv;
  row1=rowsize(argv[1]);
  column1=colsize(argv[1]);
  if (n==3) {
    rm=argv[2];
    if (!ismatrix(rm)) error(E_NOVECTOR);
    if (row1!=rowsize(rm) || column1!=colsize(rm)) error(E_VECINDEX);
  }
  else rm=makematrix(ctx,row1,column1);
  fm=rm->c.ary.entity->c.fvec.fv;
  if (flag == 0) {
    for (i=0; i<row1; i++) {
      for (j=0; j<column1; j++) {
	ii=i*column1+j;
	if (fm1[ii] >= 0.0) fm[ii] = fm1[ii];
	else fm[ii] = 0.0;
      }
    }
  } else if (flag == 1) { /* diff */
    for (i=0; i<row1; i++) {
      for (j=0; j<column1; j++) {
	ii=i*column1+j;
	if (fm1[ii] >= 0.0) fm[ii] = 1.0;
	else fm[ii] = 0.0;
      }
    }
  } else {
    error(E_MISMATCHARG);
  }
  return(rm);}

pointer MATSOFTMAX(ctx,n,argv)
register context *ctx;
int n;
register pointer argv[];
{ pointer rm;
  register int i,j,ii,row1,column1;
  register eusfloat_t *fm1,*fm;
  int flag;
  eusfloat_t *sum,sumv[10000],*max,maxv[10000],f;
  sum = sumv; max = maxv;

  ckarg2(2,3);
  flag = ckintval(argv[0]);
  if (!ismatrix(argv[1])) error(E_NOVECTOR);
  fm1=argv[1]->c.ary.entity->c.fvec.fv;
  row1=rowsize(argv[1]);
  column1=colsize(argv[1]);
  if (n==3) {
    rm=argv[2];
    if (!ismatrix(rm)) error(E_NOVECTOR);
    if (row1!=rowsize(rm) || column1!=colsize(rm)) error(E_VECINDEX);
  }
  else rm=makematrix(ctx,row1,column1);
  if (row1>10000) {
    max = (eusfloat_t *)malloc(sizeof(eusfloat_t) * row1);
    sum = (eusfloat_t *)malloc(sizeof(eusfloat_t) * row1);
    //error(E_VECINDEX);
  }
  fm=rm->c.ary.entity->c.fvec.fv;
  for (i=0; i<row1; i++) {
    ii = i*column1;
    max[i] = fm1[ii];
    for (j=1; j<column1; j++) {
      if (fm1[ii+j] > max[i])
	max[i] = fm1[ii+j];
    }
  }
  for (i=0; i<row1; i++) {
    ii = i*column1;
    sum[i] = exp(fm1[ii] - max[i]);
    for (j=1; j<column1; j++) {
      sum[i] += exp(fm1[ii+j] - max[i]);
    }
  }
  for (i=0; i<row1; i++) {
    ii = i*column1;
    for (j=0; j<column1; j++) {
      fm[ii+j] = exp(fm1[ii+j] - max[i]) / sum[i];
    }
  }
  if (flag == 1) { /* diff */
    for (i=0; i<row1; i++) {
      ii = i*column1;
      for (j=0; j<column1; j++) {
	f = fm[ii+j];
	fm[ii+j] = f * (1.0 - f);
      }
    }
  } else if(flag != 0) {
    error(E_MISMATCHARG);
  }
  if (max!=maxv) free(maxv);
  if (sum!=sumv) free(sumv);
  return(rm);}


pointer ___matprod(register context *ctx, int n, register pointer *argv)
{
  pointer mod=argv[0];
  defun(ctx,"MPROD",mod,MATPROD,NULL);
  defun(ctx,"MRELU",mod,MATRELU,NULL);
  defun(ctx,"MSOFTMAX",mod,MATSOFTMAX,NULL);}
