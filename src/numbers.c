/*-
 * Copyright (c) 2005-2019 Michael Scholz <mi-scholz@users.sourceforge.net>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * @(#)numbers.c	2.21 12/28/19
 */

#if defined(HAVE_CONFIG_H)
#include "config.h"
#endif

/* Required for C99 prototypes (log2, trunc, clog10 etc)! */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#include "fth.h"
#include "utils.h"

static FTH 	llong_tag;
static FTH 	float_tag;

#if HAVE_COMPLEX
static FTH	complex_tag;
#endif

static FTH 	bignum_tag;
static FTH 	ratio_tag;

#define FTH_MATH_ERROR_THROW(Desc)					\
	fth_throw(FTH_MATH_ERROR, "%s: %s", RUNNING_WORD(), (Desc))

#define FTH_WRONG_NUMBER_TYPE(Arg, Desc)				\
	fth_throw(FTH_WRONG_TYPE_ARG,					\
	    "%s: wrong number type, %s (%S), wanted %s",		\
	    RUNNING_WORD(),						\
	    fth_object_name(Arg),					\
	    (Arg),							\
	    (Desc))

static ficlFloat fth_pow(ficlFloat, ficlFloat);
static ficlFloat fth_floor(ficlFloat);
static ficlFloat fth_ceil(ficlFloat);
static ficlFloat fth_rint(ficlFloat);
static ficlFloat fth_trunc(ficlFloat);
static ficlFloat fth_log(ficlFloat);
static ficlFloat fth_log2(ficlFloat);
static ficlFloat fth_log10(ficlFloat);

static FTH 	make_object_number_type(const char *, fobj_t, int);
static void 	ficl_to_s(ficlVm *);

static FTH 	ll_inspect(FTH);
static FTH 	ll_to_string(FTH);
static FTH 	ll_copy(FTH);
static FTH 	ll_equal_p(FTH, FTH);

static void 	ficl_bignum_p(ficlVm *);
static void 	ficl_complex_p(ficlVm *);
static void 	ficl_even_p(ficlVm *);
static void 	ficl_exact_p(ficlVm *);
static void 	ficl_fixnum_p(ficlVm *);
static void 	ficl_inexact_p(ficlVm *);
static void 	ficl_integer_p(ficlVm *);
static void 	ficl_number_p(ficlVm *);
static void 	ficl_odd_p(ficlVm *);
static void 	ficl_llong_p(ficlVm *);
static void 	ficl_prime_p(ficlVm *);
static void 	ficl_ratio_p(ficlVm *);
static void 	ficl_unsigned_p(ficlVm *);
static void 	ficl_ullong_p(ficlVm *);

static void 	ficl_frandom(ficlVm *);
static void 	ficl_rand_seed_ref(ficlVm *);
static void 	ficl_rand_seed_set(ficlVm *);
static void 	ficl_random(ficlVm *);
static ficlFloat next_rand(void);

static void 	ficl_d_dot(ficlVm *);
static void 	ficl_d_dot_r(ficlVm *);
static void 	ficl_dot_r(ficlVm *);
static void 	ficl_u_dot_r(ficlVm *);
static void 	ficl_ud_dot(ficlVm *);
static void 	ficl_ud_dot_r(ficlVm *);

static void 	ficl_dabs(ficlVm *);
static void 	ficl_dmax(ficlVm *);
static void 	ficl_dmin(ficlVm *);
static void 	ficl_dnegate(ficlVm *);
static void 	ficl_dtwoslash(ficlVm *);
static void 	ficl_dtwostar(ficlVm *);
static void 	ficl_to_d(ficlVm *);
static void 	ficl_to_s(ficlVm *);

static char    *format_double(char *, size_t, ficlFloat);
static FTH 	fl_inspect(FTH);
static FTH 	fl_to_string(FTH);
static FTH 	fl_copy(FTH);
static FTH 	fl_equal_p(FTH, FTH);

static void 	ficl_dfloats(ficlVm *);
static void 	ficl_f_dot_r(ficlVm *);
static void 	ficl_falign(ficlVm *);
static void 	ficl_falog(ficlVm *);
static void 	ficl_fexpm1(ficlVm *);
static void 	ficl_float_p(ficlVm *);
static void 	ficl_flogp1(ficlVm *);
static void 	ficl_fsincos(ficlVm *);
static void 	ficl_inf(ficlVm *);
static void 	ficl_inf_p(ficlVm *);
static void 	ficl_nan(ficlVm *);
static void 	ficl_nan_p(ficlVm *);
static void 	ficl_to_f(ficlVm *);
static void 	ficl_uf_dot_r(ficlVm *);

static void 	ficl_cimage(ficlVm *);
static void 	ficl_creal(ficlVm *);

#if HAVE_COMPLEX
static FTH 	cp_inspect(FTH);
static FTH 	cp_to_string(FTH);
static FTH 	cp_copy(FTH);
static FTH 	cp_equal_p(FTH, FTH);

static void 	ficl_c_dot(ficlVm *);
static void 	ficl_ceq(ficlVm *);
static void 	ficl_ceqz(ficlVm *);
static void 	ficl_cnoteq(ficlVm *);
static void 	ficl_cnoteqz(ficlVm *);
static void 	ficl_creciprocal(ficlVm *);
static void 	ficl_complex_i(ficlVm *);
static void 	ficl_cnegate(ficlVm *);
static void 	ficl_make_complex_rectangular(ficlVm *);
static void 	ficl_make_complex_polar(ficlVm *);
static void 	ficl_to_c(ficlVm *);
static ficlComplex make_polar(ficlFloat, ficlFloat);
#endif				/* HAVE_COMPLEX */

static FTH 	bn_inspect(FTH);
static FTH 	bn_to_string(FTH);
static FTH 	bn_copy(FTH);
static FTH 	bn_equal_p(FTH, FTH);
static void 	bn_free(FTH);

static ficlBignum mpi_new(void);
static void 	mpi_free(ficlBignum);
static ficlBignum bn_math(FTH, FTH, int);
static FTH 	bn_add(FTH, FTH);
static FTH 	bn_sub(FTH, FTH);
static FTH 	bn_mul(FTH, FTH);
static FTH 	bn_div(FTH, FTH);
static void 	ficl_to_bn(ficlVm *);
static void 	ficl_bn_dot(ficlVm *);

static void 	ficl_bgcd(ficlVm *);
static void 	ficl_blcm(ficlVm *);
static void 	ficl_bpow(ficlVm *);
static void 	ficl_broot(ficlVm *);
static void 	ficl_bsqrt(ficlVm *);
static void 	ficl_bnegate(ficlVm *);
static void 	ficl_babs(ficlVm *);
static void 	ficl_bmin(ficlVm *);
static void 	ficl_bmax(ficlVm *);
static void 	ficl_btwostar(ficlVm *);
static void 	ficl_btwoslash(ficlVm *);
static void 	ficl_bmod(ficlVm *);
static void 	ficl_bslashmod(ficlVm *);
static void 	ficl_blshift(ficlVm *);
static void 	ficl_brshift(ficlVm *);

static FTH 	rt_inspect(FTH);
static FTH 	rt_to_string(FTH);
static FTH 	rt_copy(FTH);
static FTH 	rt_equal_p(FTH, FTH);
static void 	rt_free(FTH);

static ficlRatio mpr_new(void);
static void 	mpr_free(ficlRatio);
static FTH	make_rational(ficlBignum, ficlBignum);
static void 	ficl_to_rt(ficlVm *);
static void 	ficl_q_dot(ficlVm *);
static void 	ficl_qnegate(ficlVm *);
static void 	ficl_qfloor(ficlVm *);
static void 	ficl_qceil(ficlVm *);
static void 	ficl_qabs(ficlVm *);
static void 	ficl_qinvert(ficlVm *);
static ficlRatio rt_math(FTH, FTH, int);
static FTH 	rt_add(FTH, FTH);
static FTH 	rt_sub(FTH, FTH);
static FTH 	rt_mul(FTH, FTH);
static FTH 	rt_div(FTH, FTH);
static FTH 	number_floor(FTH);
static FTH 	number_inv(FTH);
static void 	ficl_rationalize(ficlVm *);

static void	ficl_fegetround(ficlVm *);
static void	ficl_fesetround(ficlVm *);

#define NUMB_FIXNUM_P(Obj)	(IMMEDIATE_P(Obj) && FIXNUM_P(Obj))
#define FTH_FLOAT_REF_INT(Obj)	FTH_ROUND(FTH_FLOAT_OBJECT(Obj))

#if HAVE_COMPLEX
#define FTH_COMPLEX_REAL(Obj)	creal(FTH_COMPLEX_OBJECT(Obj))
#define FTH_COMPLEX_IMAG(Obj)	cimag(FTH_COMPLEX_OBJECT(Obj))

ficlComplex
ficlStackPopComplex(ficlStack *stack)
{
	ficlComplex 	cp;

	cp = fth_complex_ref(ficl_to_fth(STACK_FTH_REF(stack)));
	stack->top--;
	return (cp);
}

void
ficlStackPushComplex(ficlStack *stack, ficlComplex cp)
{
	FTH 		fp;

	fp = fth_make_complex(cp);
	++stack->top;
	STACK_FTH_SET(stack, fp);
}
#else				/* !HAVE_COMPLEX */
#define FTH_COMPLEX_REAL(Obj)	fth_real_ref(Obj)
#define FTH_COMPLEX_IMAG(Obj)	0.0
#endif				/* HAVE_COMPLEX */

#define FTH_BIGNUM_REF_INT(Obj)  mpi_geti(FTH_BIGNUM_OBJECT(Obj))
#define FTH_BIGNUM_REF_UINT(Obj) (unsigned long)mpi_geti(FTH_BIGNUM_OBJECT(Obj))
#define FTH_BIGNUM_REF_FLOAT(Obj) mpi_getd(FTH_BIGNUM_OBJECT(Obj))
#define FTH_RATIO_REF_INT(Obj)   (long)mpr_getd(FTH_RATIO_OBJECT(Obj))
#define FTH_RATIO_REF_FLOAT(Obj) mpr_getd(FTH_RATIO_OBJECT(Obj))

/*
 * Don't forget mpi_free(bn)!
 */
ficlBignum
ficlStackPopBignum(ficlStack *stack)
{
	ficlBignum	bn;

	bn = fth_bignum_ref(ficl_to_fth(STACK_FTH_REF(stack)));
	stack->top--;
	return (bn);
}

void
ficlStackPushBignum(ficlStack *stack, ficlBignum bn)
{
	FTH 		fp;

	fp = fth_make_bignum(bn);
	++stack->top;
	STACK_FTH_SET(stack, fp);
}

/*
 * Don't forget mpr_free(rt)!
 */
ficlRatio
ficlStackPopRatio(ficlStack *stack)
{
	ficlRatio	rt;

	rt = fth_ratio_ref(ficl_to_fth(STACK_FTH_REF(stack)));
	stack->top--;
	return (rt);
}

void
ficlStackPushRatio(ficlStack *stack, ficlRatio rt)
{
	FTH 		fp;

	fp = fth_make_rational(rt);
	++stack->top;
	STACK_FTH_SET(stack, fp);
}

static FTH
make_object_number_type(const char *name, fobj_t type, int flags)
{
	FTH 		new;

	new = make_object_type(name, type);
	FTH_OBJECT_FLAG(new) = N_NUMBER_T | flags;
	return (new);
}

#if defined(HAVE_POW)
#define FTH_POW(x, y)		pow(x, y)
#else
#define FTH_POW(x, y)		FTH_NOT_IMPLEMENTED(pow)
#endif

static ficlFloat
fth_pow(ficlFloat x, ficlFloat y)
{
	return (FTH_POW(x, y));
}

#if defined(HAVE_FLOOR)
#define FTH_FLOOR(r)		floor(r)
#else
#define FTH_FLOOR(r)		((ficlFloat)((ficlInteger)(r)))
#endif

static ficlFloat
fth_floor(ficlFloat x)
{
	return (FTH_FLOOR(x));
}

#if defined(HAVE_CEIL)
#define FTH_CEIL(r)		ceil(r)
#else
#define FTH_CEIL(r)		((ficlFloat)((ficlInteger)((r) + 1.0)))
#endif

static ficlFloat
fth_ceil(ficlFloat x)
{
	return (FTH_CEIL(x));
}

#if defined(HAVE_RINT)
#define FTH_ROUND(r)		rint(r)
#else
#define FTH_ROUND(r)		fth_rint(r)
#endif

static ficlFloat
fth_rint(ficlFloat x)
{
#if defined(HAVE_RINT)
	return (rint(x));
#else
	if (x != FTH_FLOOR(x)) {
		ficlFloat 	half, half2, res;

		half = x + 0.5;
		half2 = half * 0.5;
		res = FTH_FLOOR(half);

		if (half == res && half2 != FTH_FLOOR(half2))
			return (res - 1.0);

		return (res);
	}
	return (x);
#endif
}

#if defined(HAVE_TRUNC)
#define FTH_TRUNC(r)		trunc(r)
#else
#define FTH_TRUNC(r)		(((r) < 0.0) ? -FTH_FLOOR(-(r)) : FTH_FLOOR(r))
#endif

static ficlFloat
fth_trunc(ficlFloat x)
{
	return (FTH_TRUNC(x));
}

#if defined(INFINITY)
#define FTH_INF			(ficlFloat)INFINITY
#else
static ficlFloat fth_infinity;
#define FTH_INF			fth_infinity
#endif

#if defined(NAN)
#define FTH_NAN			(ficlFloat)NAN
#else
#define FTH_NAN			sqrt(-1.0)
#endif

int
fth_isinf(ficlFloat x)
{
#if defined(HAVE_DECL_ISINF)
	return (isinf(x));
#else
	return ((x == x) && fth_isnan(x - x));
#endif
}

int
fth_isnan(ficlFloat x)
{
#if defined(HAVE_DECL_ISNAN)
	return ((int) isnan(x));
#else
	return (x != x);	/* NaN */
#endif
}

#if defined(HAVE_LOG2)
#define FTH_LOG2(r)		log2(r)
#else
#define FTH_LOG2(r)		(log10(r) / log10(2.0))
#endif

static ficlFloat
fth_log(ficlFloat x)
{
	if (x >= 0.0)
		return (log(x));
	FTH_MATH_ERROR_THROW("log, x < 0");
	/* NOTREACHED */
	return (0.0);
}

static ficlFloat
fth_log2(ficlFloat x)
{
	if (x >= 0.0)
		return (FTH_LOG2(x));
	FTH_MATH_ERROR_THROW("log2, x < 0");
	/* NOTREACHED */
	return (0.0);
}

static ficlFloat
fth_log10(ficlFloat x)
{
	if (x >= 0.0)
		return (log10(x));
	FTH_MATH_ERROR_THROW("log10, x < 0");
	/* NOTREACHED */
	return (0.0);
}

/*
 * Minix seems to lack asinh, acosh, atanh.
 */
#if !defined(HAVE_ASINH)
ficlFloat
asinh(ficlFloat x)
{
	return (log(x + sqrt(x * x + 1.0)));
}
#endif

#if !defined(HAVE_ACOSH)
ficlFloat
acosh(ficlFloat x)
{
	return (log(x + sqrt(x * x - 1.0)));
}
#endif

#if !defined(HAVE_ATANH)
ficlFloat
atanh(ficlFloat x)
{
	/* from freebsd (/usr/src/lib/msun/src/e_atanh.c) */
	if (fabs(x) > 1.0)
		return (FTH_NAN);

	if (fabs(x) == 1.0)
		return (FTH_INF);

	if (fth_isnan(x))
		return (FTH_NAN);

	return (log((1.0 + x) / (1.0 - x)) * 0.5);
}
#endif

/* === Begin of missing complex functions. === */

#if HAVE_COMPLEX

/*
 * Some libc/libm do provide them, but others do not (like FreeBSD).
 */

/* Trigonometric functions. */

#if !defined(HAVE_CSIN)
ficlComplex
csin(ficlComplex z)
{
	return (sin(creal(z)) * cosh(cimag(z)) +
	    (cos(creal(z)) * sinh(cimag(z))) * _Complex_I);
}
#endif

#if !defined(HAVE_CCOS)
ficlComplex
ccos(ficlComplex z)
{
	return (cos(creal(z)) * cosh(cimag(z)) +
	    (-sin(creal(z)) * sinh(cimag(z))) * _Complex_I);
}
#endif

#if !defined(HAVE_CTAN)
ficlComplex
ctan(ficlComplex z)
{
	return (csin(z) / ccos(z));
}
#endif

#if !defined(HAVE_CASIN)
ficlComplex
casin(ficlComplex z)
{
	return (-_Complex_I * clog(_Complex_I * z + csqrt(1.0 - z * z)));
}
#endif

#if !defined(HAVE_CACOS)
ficlComplex
cacos(ficlComplex z)
{
	return (-_Complex_I * clog(z + _Complex_I * csqrt(1.0 - z * z)));
}
#endif

#if !defined(HAVE_CATAN)
ficlComplex
catan(ficlComplex z)
{
	return (_Complex_I *
	    clog((_Complex_I + z) / (_Complex_I - z)) / 2.0);
}
#endif

#if !defined(HAVE_CATAN2)
ficlComplex
catan2(ficlComplex z, ficlComplex x)
{
	return (-_Complex_I *
	    clog((x + _Complex_I * z) / csqrt(x * x + z * z)));
}
#endif

/* Hyperbolic functions. */

#if !defined(HAVE_CSINH)
ficlComplex
csinh(ficlComplex z)
{
	return (sinh(creal(z)) * cos(cimag(z)) +
	    (cosh(creal(z)) * sin(cimag(z))) * _Complex_I);
}
#endif

#if !defined(HAVE_CCOSH)
ficlComplex
ccosh(ficlComplex z)
{
	return (cosh(creal(z)) * cos(cimag(z)) +
	    (sinh(creal(z)) * sin(cimag(z))) * _Complex_I);
}
#endif

#if !defined(HAVE_CTANH)
ficlComplex
ctanh(ficlComplex z)
{
	return (csinh(z) / ccosh(z));
}
#endif

#if !defined(HAVE_CASINH)
ficlComplex
casinh(ficlComplex z)
{
	return (clog(z + csqrt(1.0 + z * z)));
}
#endif

#if !defined(HAVE_CACOSH)
ficlComplex
cacosh(ficlComplex z)
{
	return (clog(z + csqrt(z * z - 1.0)));
}
#endif

#if !defined(HAVE_CATANH)
ficlComplex
catanh(ficlComplex z)
{
	return (clog((1.0 + z) / (1.0 - z)) / 2.0);
}
#endif

/* Exponential and logarithmic functions. */

#if !defined(HAVE_CEXP)
ficlComplex
cexp(ficlComplex z)
{
	return (exp(creal(z)) * cos(cimag(z)) +
	    (exp(creal(z)) * sin(cimag(z))) * _Complex_I);
}
#endif

#if !defined(HAVE_CLOG)
ficlComplex
clog(ficlComplex z)
{
	return (log(fabs(cabs(z))) + carg(z) * _Complex_I);
}
#endif

#if !defined(HAVE_CLOG10)
ficlComplex
clog10(ficlComplex z)
{
	return (clog(z) / log(10));
}
#endif

/* Power functions. */

#if !defined(HAVE_CPOW)
ficlComplex
cpow(ficlComplex a, ficlComplex z)
{
	/* from netbsd (/usr/src/lib/libm/complex/cpow.c) */
	double 		x, y, r, theta, absa, arga;

	x = creal(z);
	y = cimag(z);
	absa = cabs(a);

	if (absa == 0.0)
		return (0.0 + 0.0 * _Complex_I);

	arga = carg(a);
	r = FTH_POW(absa, x);
	theta = x * arga;

	if (y != 0.0) {
		r = r * exp(-y * arga);
		theta = theta + y * log(absa);
	}
	return (r * cos(theta) + (r * sin(theta)) * _Complex_I);
}
#endif

#if !defined(HAVE_CSQRT)
ficlComplex
csqrt(ficlComplex z)
{
	ficlFloat 	r, x;

	if (cimag(z) < 0.0)
		return (conj(csqrt(conj(z))));

	r = cabs(z);
	x = creal(z);
	return (sqrt((r + x) / 2.0) + sqrt((r - x) / 2.0) * _Complex_I);
}
#endif

/* Absolute value and conjugates. */

#if !defined(HAVE_CABS)
ficlFloat
cabs(ficlComplex z)
{
	return (hypot(creal(z), cimag(z)));
}
#endif

#if !defined(HAVE_CABS2)
ficlFloat
cabs2(ficlComplex z)
{
	return (creal(z) * creal(z) + cimag(z) * cimag(z));
}
#endif

#if !defined(HAVE_CARG)
ficlFloat
carg(ficlComplex z)
{
	return (atan2(cimag(z), creal(z)));
}
#endif

#if !defined(HAVE_CONJ)
ficlComplex
conj(ficlComplex z)
{
	return (~z);
}
#endif
#endif				/* HAVE_COMPLEX */

/* End of missing complex functions. */

/*
 * ficlFloat
 * ficlComplex
 */
#define N_FUNC_ONE_ARG(Name, CName, Type)				\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	ficl ## Type x;							\
									\
	FTH_STACK_CHECK(vm, 1, 1);					\
	x = ficlStackPop ## Type(vm->dataStack);			\
	ficlStackPush ## Type(vm->dataStack, CName(x));			\
}									\
static char* h_ ## Name = "( x -- y )  y = " #CName "(x)"

/*
 * ficlFloat
 * ficlComplex
 */
#define N_FUNC_TWO_ARGS(Name, CName, Type)				\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	ficl ## Type x;							\
	ficl ## Type y;							\
									\
	FTH_STACK_CHECK(vm, 2, 1);					\
	y = ficlStackPop ## Type(vm->dataStack);			\
	x = ficlStackPop ## Type(vm->dataStack);			\
	ficlStackPush ## Type(vm->dataStack, CName(x, y));		\
}									\
static char* h_ ## Name = "( x y -- z )  z = " #CName "(x, y)"

/*
 * ficl2Integer
 * ficlComplex
 */
#define N_FUNC_TWO_ARGS_OP(Name, OP, Type)				\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	ficl ## Type x;							\
	ficl ## Type y;							\
									\
	FTH_STACK_CHECK(vm, 2, 1);					\
	y = ficlStackPop ## Type(vm->dataStack);			\
	x = ficlStackPop ## Type(vm->dataStack);			\
	ficlStackPush ## Type(vm->dataStack, x OP y);			\
}									\
static char* h_ ## Name = "( x y -- z )  z = x " #OP " y"

/*
 * ficl2Integer
 */
#define N_FUNC_TEST_ZERO(Name, OP, Type)				\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	ficl ## Type x;							\
									\
	FTH_STACK_CHECK(vm, 1, 1);					\
	x = ficlStackPop ## Type(vm->dataStack);			\
	ficlStackPushBoolean(vm->dataStack, (x OP 0));			\
}									\
static char* h_ ## Name = "( x -- f )  x " #OP " 0 => flag"

/*
 * ficlUnsigned
 * ficl2Integer
 * ficl2Unsigned
 */
#define N_FUNC_TEST_TWO_OP(Name, OP, Type)				\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	ficl ## Type x;							\
	ficl ## Type y;							\
									\
	FTH_STACK_CHECK(vm, 2, 1);					\
	y = ficlStackPop ## Type(vm->dataStack);			\
	x = ficlStackPop ## Type(vm->dataStack);			\
	ficlStackPushBoolean(vm->dataStack, (x OP y));			\
}									\
static char* h_ ## Name = "( x y -- f )  x " #OP " y => flag"

static void
ficl_to_s(ficlVm *vm)
{
#define h_to_s "( x -- y )  Convert any number X to ficlInteger"
	FTH 		n;

	FTH_STACK_CHECK(vm, 1, 1);
	n = fth_pop_ficl_cell(vm);
	ficlStackPushInteger(vm->dataStack, fth_int_ref(n));
}

static void
ficl_to_d(ficlVm *vm)
{
#define h_to_d "( x -- d )  Convert any number X to ficl2Integer"
	ficl2Integer 	d;

	FTH_STACK_CHECK(vm, 1, 1);
	d = ficlStackPop2Integer(vm->dataStack);
	ficlStackPushFTH(vm->dataStack, fth_make_llong(d));
}

static void
ficl_to_ud(ficlVm *vm)
{
#define h_to_ud "( x -- ud )  Convert any number X to ficl2Unsigned"
	ficl2Unsigned	ud;

	FTH_STACK_CHECK(vm, 1, 1);
	ud = ficlStackPop2Unsigned(vm->dataStack);
	ficlStackPushFTH(vm->dataStack, fth_make_ullong(ud));
}

static void
ficl_to_f(ficlVm *vm)
{
#define h_to_f "( x -- y )  Convert any number X to ficlFloat"
	ficlFloat	f;

	FTH_STACK_CHECK(vm, 1, 1);
	f = ficlStackPopFloat(vm->dataStack);
	ficlStackPushFloat(vm->dataStack, f);
}

#if HAVE_COMPLEX
static void
ficl_to_c(ficlVm *vm)
{
#define h_to_c "( x -- y )  Convert any number X to ficlComplex"
	ficlComplex 	cp;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);
	ficlStackPushComplex(vm->dataStack, cp);
}
#endif

static void
ficl_to_bn(ficlVm *vm)
{
#define h_to_bn "( x -- y )  Convert any number X to ficlBignum"
	ficlBignum	bn;

	FTH_STACK_CHECK(vm, 1, 1);
	bn = ficlStackPopBignum(vm->dataStack);
	ficlStackPushBignum(vm->dataStack, bn);
}

static void
ficl_to_rt(ficlVm *vm)
{
#define h_to_rt "( x -- y )  Convert any number X to ficlRatio"
	ficlRatio	rt;

	FTH_STACK_CHECK(vm, 1, 1);
	rt = ficlStackPopRatio(vm->dataStack);
	ficlStackPushRatio(vm->dataStack, rt);
}

/* === LONG-LONG === */

#define h_list_of_llong_functions "\
*** NUMBER PRIMITIVES ***\n\
number?   fixnum?   unsigned?\n\
long-long? ulong-long?\n\
integer?  exact?    inexact?\n\
make-long-long     make-ulong-long\n\
rand-seed-ref  rand-seed-set!\n\
random    frandom\n\
.r   u.r  d.   ud.  d.r  ud.r\n\
u=   u<>  u<   u>   u<=  u>=\n\
s>d  s>ud d>s  f>d  f>ud d>f\n\
d0=  (dzero?)  d0<> d0<  (dnegative?) d0>  d0<= d0>= (dpositive?)\n\
d=   d<>  d<   d>   d<=  d>=\n\
du=  du<> du<  du>  du<= du>=\n\
d+   d-   d*   d/\n\
dnegate   dabs dmin dmax d2*  d2/"

static FTH
ll_inspect(FTH self)
{
	return (fth_make_string_format("%s: %lld",
		FTH_INSTANCE_NAME(self), FTH_LONG_OBJECT(self)));
}

static FTH
ll_to_string(FTH self)
{
	return (fth_make_string_format("%lld", FTH_LONG_OBJECT(self)));
}

static FTH
ll_copy(FTH self)
{
	return (fth_make_llong(FTH_LONG_OBJECT(self)));
}

static FTH
ll_equal_p(FTH self, FTH obj)
{
	return (BOOL_TO_FTH(FTH_LONG_OBJECT(self) == FTH_LONG_OBJECT(obj)));
}

FTH
fth_make_llong(ficl2Integer d)
{
	FTH 		self;

	self = fth_make_instance(llong_tag, NULL);
	FTH_LONG_OBJECT_SET(self, d);
	return (self);
}

FTH
fth_make_ullong(ficl2Unsigned ud)
{
	FTH 		self;

	self = fth_make_instance(llong_tag, NULL);
	FTH_ULONG_OBJECT_SET(self, ud);
	return (self);
}

FTH
fth_llong_copy(FTH obj)
{
	if (FTH_LLONG_P(obj))
		return (ll_copy(obj));
	return (obj);
}

int
fth_fixnum_p(FTH obj)
{
	return (NUMB_FIXNUM_P(obj));
}

int
fth_number_p(FTH obj)
{
	return (NUMB_FIXNUM_P(obj) || FTH_NUMBER_T_P(obj));
}

int
fth_exact_p(FTH obj)
{
	return (NUMB_FIXNUM_P(obj) || FTH_EXACT_T_P(obj));
}

int
fth_integer_p(FTH obj)
{
	return (NUMB_FIXNUM_P(obj) || FTH_LLONG_P(obj));
}

int
fth_char_p(FTH obj)
{
	return (NUMB_FIXNUM_P(obj) && isprint((int) FIX_TO_INT(obj)));
}

int
fth_unsigned_p(FTH obj)
{
	return (fth_integer_p(obj) && fth_long_long_ref(obj) >= 0);
}

int
fth_ullong_p(FTH obj)
{
	return (FTH_LLONG_P(obj) && FTH_LONG_OBJECT(obj) >= 0);
}

static void
ficl_number_p(ficlVm *vm)
{
#define h_number_p "( obj -- f )  test if OBJ is a number\n\
nil number? => #f\n\
0   number? => #t\n\
0i  number? => #t\n\
Return #t if OBJ is a number (ficlInteger, ficl2Integer, ficlFloat, \
ficlComplex, ficlBignum, ficlRatio), otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_NUMBER_P(obj));
}

static void
ficl_fixnum_p(ficlVm *vm)
{
#define h_fixnum_p "( obj -- f )  test if OBJ is fixnum\n\
nil fixnum? => #f\n\
0   fixnum? => #t\n\
0x3fffffff    fixnum? => #t\n\
0x3fffffff 1+ fixnum? => #f\n\
Return #t if OBJ is fixnum (ficlInteger/ficlUnsigned), otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, NUMB_FIXNUM_P(obj));
}

static void
ficl_unsigned_p(ficlVm *vm)
{
#define h_unsigned_p "( obj -- f )  test if OBJ is unsigned integer\n\
nil unsigned? => #f\n\
-1  unsigned? => #f\n\
0   unsigned? => #t\n\
0xffffffffffff unsigned? => #t\n\
Return #t if OBJ is unsigned integer (ficlUnsigned, \
ficl2Unsigned, ficlBignum), otherwise #f."
	FTH 		obj;
	int 		flag;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);

	if (FTH_UNSIGNED_P(obj))
		flag = 1;
	else if (FTH_BIGNUM_P(obj))
		flag = (mpi_sgn(FTH_BIGNUM_OBJECT(obj)) >= 0);
	else
		flag = 0;

	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_llong_p(ficlVm *vm)
{
#define h_llong_p "( obj -- f )  test if OBJ is long-long integer\n\
nil long-long? => #f\n\
-1  long-long? => #f\n\
-1 make-long-long long-long? => #t\n\
Return #t if OBJ is long-long object (ficl2Integer/ficl2Unsigned), \
otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_LLONG_P(obj));
}

static void
ficl_ullong_p(ficlVm *vm)
{
#define h_ullong_p "( obj -- f )  test if OBJ is unsigned long-long integer\n\
nil ulong-long? => #f\n\
1   ulong-long? => #f\n\
1 make-ulong-long ulong-long? => #t\n\
Return #t if OBJ is ulong-long object (ficl2Unsigned), otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_ULLONG_P(obj));
}

static void
ficl_integer_p(ficlVm *vm)
{
#define h_integer_p "( obj -- f )  test if OBJ is an integer\n\
nil integer? => #f\n\
1.0 integer? => #f\n\
-1  integer? => #t\n\
1 make-long-long integer? => #t\n\
12345678901234567890 integer? => #t\n\
Return #t if OBJ is an integer (ficlInteger, ficl2Integer, or ficlBignum), \
otherwise #f."
	FTH 		obj;
	int		flag;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	flag = FTH_INTEGER_P(obj) || FTH_BIGNUM_P(obj);
	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_exact_p(ficlVm *vm)
{
#define h_exact_p "( obj -- f )  test if OBJ is an exact number\n\
1   exact? => #t\n\
1/2 exact? => #t\n\
1.0 exact? => #f\n\
1i  exact? => #f\n\
Return #t if OBJ is an exact number (not ficlFloat or \
ficlComplex), otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_EXACT_P(obj));
}

static void
ficl_inexact_p(ficlVm *vm)
{
#define h_inexact_p "( obj -- f )  test if OBJ is an inexact number\n\
1.0 inexact? => #t\n\
1i  inexact? => #t\n\
1   inexact? => #f\n\
1/2 inexact? => #f\n\
Return #t if OBJ is an inexact number (ficlFloat, ficlComplex), otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_INEXACT_P(obj));
}

/*
 * Integer types.
 *
 * Return a FTH fixnum or a FTH llong object depending on N.
 */
FTH
fth_make_int(ficlInteger n)
{
	if (FIXABLE_P(n))
		return (INT_TO_FIX(n));
	return (fth_make_llong((ficl2Integer) n));
}

/*
 * Return a FTH unsigned fixnum or a FTH ullong object depending on U.
 */
FTH
fth_make_unsigned(ficlUnsigned u)
{
	if (UFIXABLE_P(u))
		return (UNSIGNED_TO_FIX(u));
	return (fth_make_ullong((ficl2Unsigned) u));
}

/*
 * Return a FTH fixnum or a FTH llong object depending on D.
 */
FTH
fth_make_long_long(ficl2Integer d)
{
	if (FIXABLE_P(d))
		return (INT_TO_FIX((ficlInteger) d));
	return (fth_make_llong(d));
}

/*
 * Return a FTH unsigned fixnum or a FTH ullong object depending on UD.
 */
FTH
fth_make_ulong_long(ficl2Unsigned ud)
{
	if (UFIXABLE_P(ud))
		return (UNSIGNED_TO_FIX((ficlUnsigned) ud));
	return (fth_make_ullong(ud));
}

/*
 * Supposed to be used in FTH_INT_REF() macro.
 */
ficlInteger
fth_integer_ref(FTH x)
{
	if (NUMB_FIXNUM_P(x))
		return (FIX_TO_INT(x));

	if (FTH_LLONG_P(x))
		return (ficlInteger) (FTH_LONG_OBJECT(x));

	return ((ficlInteger) x);
}

/*
 * Convert any number to type.
 *
 * Return C ficlInteger from OBJ.
 */
ficlInteger
fth_int_ref(FTH obj)
{
	ficlInteger 	i;

	if (NUMB_FIXNUM_P(obj))
		return (FIX_TO_INT(obj));

	if (!FTH_NUMBER_T_P(obj))
		FTH_WRONG_NUMBER_TYPE(obj, "a ficlInteger");

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_LLONG_T:
		i = (ficlInteger) FTH_LONG_OBJECT(obj);
		break;
	case FTH_FLOAT_T:
		i = (ficlInteger) FTH_FLOAT_REF_INT(obj);
		break;
	case FTH_RATIO_T:
		i = FTH_RATIO_REF_INT(obj);
		break;
	case FTH_BIGNUM_T:
		i = FTH_BIGNUM_REF_INT(obj);
		break;
	case FTH_COMPLEX_T:
	default:
		i = (ficlInteger) FTH_ROUND(FTH_COMPLEX_REAL(obj));
		break;
	}

	return (i);
}

/*
 * Return C ficlInteger from OBJ.  If OBJ doesn't fit in Fixnum, FTH llong,
 * FTH float, FTH complex, or any bignum, return fallback.
 */
ficlInteger
fth_int_ref_or_else(FTH obj, ficlInteger fallback)
{
	ficlInteger 	i;

	if (NUMB_FIXNUM_P(obj))
		return (FIX_TO_INT(obj));

	if (!FTH_NUMBER_T_P(obj))
		return (fallback);

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_LLONG_T:
		i = (ficlInteger) FTH_LONG_OBJECT(obj);
		break;
	case FTH_FLOAT_T:
		i = (ficlInteger) FTH_FLOAT_REF_INT(obj);
		break;
	case FTH_RATIO_T:
		i = FTH_RATIO_REF_INT(obj);
		break;
	case FTH_BIGNUM_T:
		i = FTH_BIGNUM_REF_INT(obj);
		break;
	case FTH_COMPLEX_T:
	default:
		i = (ficlInteger) FTH_ROUND(FTH_COMPLEX_REAL(obj));
		break;
	}

	return (i);
}

/*
 * Return C ficl2Integer from OBJ.
 */
ficl2Integer
fth_long_long_ref(FTH obj)
{
	ficl2Integer 	d;

	if (FTH_LLONG_P(obj))
		return (FTH_LONG_OBJECT(obj));

	if (NUMB_FIXNUM_P(obj))
		return ((ficl2Integer) FIX_TO_INT(obj));

	if (!FTH_NUMBER_T_P(obj))
		FTH_WRONG_NUMBER_TYPE(obj, "a ficl2Integer");

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_FLOAT_T:
		d = (ficl2Integer) FTH_FLOAT_REF_INT(obj);
		break;
	case FTH_RATIO_T:
		d = (ficl2Integer) FTH_RATIO_REF_INT(obj);
		break;
	case FTH_BIGNUM_T:
		d = (ficl2Integer) FTH_BIGNUM_REF_INT(obj);
		break;
	case FTH_COMPLEX_T:
	default:
		d = (ficl2Integer) FTH_ROUND(FTH_COMPLEX_REAL(obj));
		break;
	}

	return (d);
}

/*
 * Return C ficlUnsigned from OBJ.
 */
ficlUnsigned
fth_unsigned_ref(FTH obj)
{
	ficlUnsigned 	u;

	if (NUMB_FIXNUM_P(obj))
		return (FIX_TO_UNSIGNED(obj));

	if (!FTH_NUMBER_T_P(obj))
		FTH_WRONG_NUMBER_TYPE(obj, "a ficlUnsigned");

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_LLONG_T:
		u = (ficlUnsigned) FTH_LONG_OBJECT(obj);
		break;
	case FTH_FLOAT_T:
		u = (ficlUnsigned) FTH_FLOAT_REF_INT(obj);
		break;
	case FTH_RATIO_T:
		u = (ficlUnsigned) FTH_RATIO_REF_INT(obj);
		break;
	case FTH_BIGNUM_T:
		u = (ficlUnsigned) FTH_BIGNUM_REF_UINT(obj);
		break;
	case FTH_COMPLEX_T:
	default:
		u = (ficlUnsigned) FTH_ROUND(FTH_COMPLEX_REAL(obj));
		break;
	}

	return (u);
}

/*
 * Return C ficl2Unsigned from OBJ.
 */
ficl2Unsigned
fth_ulong_long_ref(FTH obj)
{
	ficl2Unsigned 	ud;

	if (FTH_ULLONG_P(obj))
		return (FTH_ULONG_OBJECT(obj));

	if (NUMB_FIXNUM_P(obj))
		return ((ficl2Unsigned) FIX_TO_UNSIGNED(obj));

	if (!FTH_NUMBER_T_P(obj))
		FTH_WRONG_NUMBER_TYPE(obj, "a ficl2Unsigned");

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_FLOAT_T:
		ud = (ficl2Unsigned) FTH_FLOAT_REF_INT(obj);
		break;
	case FTH_RATIO_T:
		ud = (ficl2Unsigned) FTH_RATIO_REF_INT(obj);
		break;
	case FTH_BIGNUM_T:
		ud = (ficl2Unsigned) FTH_BIGNUM_REF_UINT(obj);
		break;
	case FTH_COMPLEX_T:
	default:
		ud = (ficl2Unsigned) FTH_ROUND(FTH_COMPLEX_REAL(obj));
		break;
	}

	return (ud);
}

/*
 * Return C ficlFloat from OBJ.  If OBJ isn't of type Fixnum, FTH llong,
 * FTH float, FTH complex, or any bignum, throw an exception.
 */
ficlFloat
fth_float_ref(FTH obj)
{
	ficlFloat 	f;

	if (FTH_FLOAT_T_P(obj))
		return (FTH_FLOAT_OBJECT(obj));

	if (NUMB_FIXNUM_P(obj))
		return ((ficlFloat) FIX_TO_INT(obj));

	if (!FTH_NUMBER_T_P(obj))
		FTH_WRONG_NUMBER_TYPE(obj, "a ficlFloat");

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_COMPLEX_T:
		f = FTH_COMPLEX_REAL(obj);
		break;
	case FTH_RATIO_T:
		f = FTH_RATIO_REF_FLOAT(obj);
		break;
	case FTH_BIGNUM_T:
		f = FTH_BIGNUM_REF_FLOAT(obj);
		break;
	case FTH_LLONG_T:
	default:
		f = (ficlFloat) FTH_LONG_OBJECT(obj);
		break;
	}

	return (f);
}

/*
 * Alias for fth_float_ref().
 */
ficlFloat
fth_real_ref(FTH x)
{
	return (fth_float_ref(x));
}

ficlFloat
fth_float_ref_or_else(FTH obj, ficlFloat fallback)
{
	ficlFloat 	f;

	if (FTH_FLOAT_T_P(obj))
		return (FTH_FLOAT_OBJECT(obj));

	if (NUMB_FIXNUM_P(obj))
		return ((ficlFloat) FIX_TO_INT(obj));

	if (!FTH_NUMBER_T_P(obj))
		return (fallback);

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_COMPLEX_T:
		f = FTH_COMPLEX_REAL(obj);
		break;
	case FTH_RATIO_T:
		f = FTH_RATIO_REF_FLOAT(obj);
		break;
	case FTH_BIGNUM_T:
		f = FTH_BIGNUM_REF_FLOAT(obj);
		break;
	case FTH_LLONG_T:
	default:
		f = (ficlFloat) FTH_LONG_OBJECT(obj);
		break;
	}

	return (f);
}

#if HAVE_COMPLEX
/*
 * Return C ficlComplex from OBJ.
 */
ficlComplex
fth_complex_ref(FTH obj)
{
	if (FTH_COMPLEX_P(obj))
		return (FTH_COMPLEX_OBJECT(obj));
	return (fth_float_ref(obj) + 0.0 * _Complex_I);
}
#endif

/*
 * Don't forget mpi_free(bn)!
 */
ficlBignum
fth_bignum_ref(FTH obj)
{
	ficlBignum	bn;

	bn = mpi_new();

	if (!FTH_NUMBER_T_P(obj)) {
		mpi_seti(bn, fth_integer_ref(obj));
		return (bn);
	}

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_BIGNUM_T:
		mpi_set(bn, FTH_BIGNUM_OBJECT(obj));
		break;
	case FTH_FLOAT_T:
		mpi_setd(bn, FTH_FLOAT_OBJECT(obj));
		break;
	case FTH_RATIO_T:
		mpi_setd(bn, FTH_RATIO_REF_FLOAT(obj));
		break;
	case FTH_LLONG_T:
		mpi_seti(bn, FTH_LONG_OBJECT(obj));
		break;
	case FTH_COMPLEX_T:
		mpi_setd(bn, FTH_COMPLEX_REAL(obj));
		break;
	default:
		mpi_seti(bn, fth_integer_ref(obj));
		break;
	}

	return (bn);
}

/*
 * Don't forget mpr_free(rt)!
 */
ficlRatio
fth_ratio_ref(FTH obj)
{
	ficlRatio	rt;

	rt = mpr_new();

	if (!FTH_NUMBER_T_P(obj)) {
		mpr_seti(rt, fth_integer_ref(obj), 1);
		return (rt);
	}

	switch (FTH_INSTANCE_TYPE(obj)) {
	case FTH_RATIO_T:
		mpr_set(rt, FTH_RATIO_OBJECT(obj));
		break;
	case FTH_FLOAT_T:
		mpr_setd(rt, FTH_FLOAT_OBJECT(obj));
		break;
	case FTH_BIGNUM_T:
		mpi_set(mpr_num(rt), FTH_BIGNUM_OBJECT(obj));
		mpi_seti(mpr_den(rt), 1);
		break;
	case FTH_LLONG_T:
		mpr_seti(rt, (long)FTH_LONG_OBJECT(obj), 1);
		break;
	case FTH_COMPLEX_T:
		mpr_setd(rt, FTH_COMPLEX_REAL(obj));
		break;
	default:
		mpr_seti(rt, fth_integer_ref(obj), 1);
		break;
	}

	return (rt);
}

/* === RANDOM === */

/*
 * From clm/cmus.c.
 */
static ficlUnsigned fth_rand_rnd;

#define INVERSE_MAX_RAND	0.0000610351563
#define INVERSE_MAX_RAND2	0.000030517579

void
fth_srand(ficlUnsigned val)
{
	fth_rand_rnd = val;
}

static ficlFloat
next_rand(void)
{
	unsigned long 	val;
	fth_rand_rnd = fth_rand_rnd * 1103515245 + 12345;
	val = (unsigned long) (fth_rand_rnd >> 16) & 32767;
	return ((ficlFloat) val);
}

/* -amp to amp as double */
ficlFloat
fth_frandom(ficlFloat amp)
{
	return (amp * (next_rand() * INVERSE_MAX_RAND - 1.0));
}

/* 0..amp as double */
ficlFloat
fth_random(ficlFloat amp)
{
	return (amp * (next_rand() * INVERSE_MAX_RAND2));
}

static void
ficl_random(ficlVm *vm)
{
#define h_random "( r -- 0.0..r)  return randomized value\n\
1 random => 0.513855\n\
Return pseudo randomized value between 0.0 and R.\n\
See also frandom."
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 1);
	f = fth_random(ficlStackPopFloat(vm->dataStack));
	ficlStackPushFloat(vm->dataStack, f);
}

static void
ficl_frandom(ficlVm *vm)
{
#define h_frandom "( r -- -r...r)  return randomized value\n\
1 frandom => -0.64856\n\
Return pseudo randomized value between -R and R.\n\
See also random."
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 1);
	f = fth_frandom(ficlStackPopFloat(vm->dataStack));
	ficlStackPushFloat(vm->dataStack, f);
}

static void
ficl_rand_seed_ref(ficlVm *vm)
{
#define h_rand_seed_ref "( -- seed )  return rand seed\n\
rand-seed-ref => 213\n\
Return content of the seed variable fth_rand_rnd.\n\
See also rand-seed-set!."
	FTH_STACK_CHECK(vm, 0, 1);
	ficlStackPushUnsigned(vm->dataStack, fth_rand_rnd);
}

static void
ficl_rand_seed_set(ficlVm *vm)
{
#define h_rand_seed_set "( seed -- )  set rand seed\n\
213 rand-seed-set!\n\
Set SEED to the seed variable fth_rand_rnd.\n\
See also rand-seed-ref."
	FTH_STACK_CHECK(vm, 1, 0);
	fth_rand_rnd = ficlStackPopUnsigned(vm->dataStack);
}

/* === FORMATTED NUMBER OUTPUT === */

static void
ficl_dot_r(ficlVm *vm)
{
#define h_dot_r "( n1 n2 -- )  formatted number output\n\
17 3 .r => | 17 |\n\
Print integer N1 in a right-adjusted field of N2 characters.\n\
See also u.r"
	ficlInteger 	n1;
	int 		n2;

	FTH_STACK_CHECK(vm, 2, 0);
	n2 = (int) ficlStackPopInteger(vm->dataStack);
	n1 = ficlStackPopInteger(vm->dataStack);
	fth_printf("%*ld ", n2, n1);
}

static void
ficl_u_dot_r(ficlVm *vm)
{
#define h_u_dot_r "( u n -- )  formatted number output\n\
17 3 u.r => | 17 |\n\
Print unsigned integer U in a right-adjusted field of N characters.\n\
See also .r"
	ficlUnsigned 	u;
	int 		n;

	FTH_STACK_CHECK(vm, 2, 0);
	n = (int) ficlStackPopInteger(vm->dataStack);
	u = ficlStackPopUnsigned(vm->dataStack);
	fth_printf("%*lu ", n, u);
}

static void
ficl_d_dot(ficlVm *vm)
{
#define h_d_dot "( d -- )  number output\n\
17 d. => 17\n\
Print (Forth) double D (ficl2Integer).\n\
See also ud."
	FTH_STACK_CHECK(vm, 1, 0);
	fth_printf("%lld ", ficlStackPop2Integer(vm->dataStack));
}

static void
ficl_ud_dot(ficlVm *vm)
{
#define h_ud_dot "( ud -- )  number output\n\
17 ud. => 17\n\
Print (Forth) unsigned double UD (ficl2Unsigned).\n\
See also d."
	FTH_STACK_CHECK(vm, 1, 0);
	fth_printf("%llu ", ficlStackPop2Unsigned(vm->dataStack));
}

static void
ficl_d_dot_r(ficlVm *vm)
{
#define h_d_dot_r "( d n -- )  formatted number output\n\
17 3 d.r => | 17 |\n\
Print (Forth) double D (ficl2Integer) \
in a right-adjusted field of N characters.\n\
See also ud.r"
	ficl2Integer 	d;
	int 		n;

	FTH_STACK_CHECK(vm, 2, 0);
	n = (int) ficlStackPopInteger(vm->dataStack);
	d = ficlStackPop2Integer(vm->dataStack);
	fth_printf("%*lld ", n, d);
}

static void
ficl_ud_dot_r(ficlVm *vm)
{
#define h_ud_dot_r "( ud n -- )  formatted number output\n\
17 3 ud.r => | 17 |\n\
Print (Forth) unsigned double UD (ficl2Unsigned) \
in a right-adjusted field of N characters.\n\
See also d.r"
	ficl2Unsigned 	ud;
	int 		n;

	FTH_STACK_CHECK(vm, 2, 0);
	n = (int) ficlStackPopInteger(vm->dataStack);
	ud = ficlStackPop2Unsigned(vm->dataStack);
	fth_printf("%*llu ", n, ud);
}

static void
ficl_dnegate(ficlVm *vm)
{
#define h_dnegate "( x -- y )  y = -x"
	ficl2Integer	x;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPop2Integer(vm->dataStack);
	ficlStackPush2Integer(vm->dataStack, -x);
}

static void
ficl_dabs(ficlVm *vm)
{
#define h_dabs "( x -- y )  y = abs(x)"
	ficl2Integer 	x;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPop2Integer(vm->dataStack);
	ficlStackPush2Integer(vm->dataStack, (x < 0) ? -x : x);
}

static void
ficl_dmin(ficlVm *vm)
{
#define h_dmin "( x y -- z )  z = min(x, y)"
	ficl2Integer 	x;
	ficl2Integer 	y;

	FTH_STACK_CHECK(vm, 2, 1);
	y = ficlStackPop2Integer(vm->dataStack);
	x = ficlStackPop2Integer(vm->dataStack);
	ficlStackPush2Integer(vm->dataStack, (x < y) ? x : y);
}

static void
ficl_dmax(ficlVm *vm)
{
#define h_dmax "( x y -- z )  z = max(x, y)"
	ficl2Integer 	x;
	ficl2Integer 	y;

	FTH_STACK_CHECK(vm, 2, 1);
	y = ficlStackPop2Integer(vm->dataStack);
	x = ficlStackPop2Integer(vm->dataStack);
	ficlStackPush2Integer(vm->dataStack, (x > y) ? x : y);
}

static void
ficl_dtwostar(ficlVm *vm)
{
#define h_dtwostar "( x -- y )  y = x * 2"
	ficl2Integer	x;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPop2Integer(vm->dataStack);
	ficlStackPush2Integer(vm->dataStack, x * 2);
}

static void
ficl_dtwoslash(ficlVm *vm)
{
#define h_dtwoslash "( x -- y )  y = x / 2"
	ficl2Integer	x;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPop2Integer(vm->dataStack);
	ficlStackPush2Integer(vm->dataStack, x / 2);
}

N_FUNC_TEST_TWO_OP(ueq, ==, Unsigned);
N_FUNC_TEST_TWO_OP(unoteq, !=, Unsigned);
N_FUNC_TEST_TWO_OP(uless, <, Unsigned);
N_FUNC_TEST_TWO_OP(ulesseq, <=, Unsigned);
N_FUNC_TEST_TWO_OP(ugreater, >, Unsigned);
N_FUNC_TEST_TWO_OP(ugreatereq, >=, Unsigned);

N_FUNC_TEST_ZERO(dzero, ==, 2Integer);
N_FUNC_TEST_ZERO(dnotz, !=, 2Integer);
N_FUNC_TEST_ZERO(dlessz, <, 2Integer);
N_FUNC_TEST_ZERO(dlesseqz, <=, 2Integer);
N_FUNC_TEST_ZERO(dgreaterz, >, 2Integer);
N_FUNC_TEST_ZERO(dgreatereqz, >=, 2Integer);

N_FUNC_TEST_TWO_OP(deq, ==, 2Integer);
N_FUNC_TEST_TWO_OP(dnoteq, !=, 2Integer);
N_FUNC_TEST_TWO_OP(dless, <, 2Integer);
N_FUNC_TEST_TWO_OP(dlesseq, <=, 2Integer);
N_FUNC_TEST_TWO_OP(dgreater, >, 2Integer);
N_FUNC_TEST_TWO_OP(dgreatereq, >=, 2Integer);

N_FUNC_TEST_TWO_OP(dueq, ==, 2Unsigned);
N_FUNC_TEST_TWO_OP(dunoteq, !=, 2Unsigned);
N_FUNC_TEST_TWO_OP(duless, <, 2Unsigned);
N_FUNC_TEST_TWO_OP(dulesseq, <=, 2Unsigned);
N_FUNC_TEST_TWO_OP(dugreater, >, 2Unsigned);
N_FUNC_TEST_TWO_OP(dugreatereq, >=, 2Unsigned);

N_FUNC_TWO_ARGS_OP(dadd, +, 2Integer);
N_FUNC_TWO_ARGS_OP(dsub, -, 2Integer);
N_FUNC_TWO_ARGS_OP(dmul, *, 2Integer);
N_FUNC_TWO_ARGS_OP(ddiv, /, 2Integer);

/* === FLOAT === */

#define h_list_of_float_functions "\
*** FLOAT PRIMITIVES ***\n\
float?    inf?      nan?\n\
inf       nan\n\
f.r       uf.r\n\
floats    (sfloats and dfloats)\n\
falign    f>s       s>f\n\
f**       (fpow)    fabs\n\
fmod      floor     fceil     ftrunc\n\
fround    fsqrt     fexp      fexpm1\n\
flog      flogp1    (flog1p)  flog2     flog10    falog\n\
fsin      fcos      ftan      fsincos\n\
fasin     facos     fatan     fatan2\n\
fsinh     fcosh     ftanh\n\
fasinh    facosh    fatanh\n\
*** FLOAT CONSTANTS ***\n\
euler  ln-two  ln-ten  pi  two-pi  half-pi  sqrt-two"

static char    *
format_double(char *buf, size_t size, ficlFloat f)
{
	int 		i;
	int 		len;
	int 		isize;
	int 		okay;

	len = snprintf(buf, size, "%g", f);
	okay = 0;

	for (i = 0; i < len; i++) {
		if (buf[i] == 'e' || buf[i] == '.') {
			okay = 1;
			break;
		}
	}

	isize = (int) size;

	if (!okay && (len + 2) < isize)
		buf[len] = '.';

	buf[len + 1] = '0';
	buf[len + 2] = '\0';
	return (buf);
}

static char 	numbers_scratch[BUFSIZ];

static FTH
fl_inspect(FTH self)
{
	ficlFloat 	f;
	FTH 		fs;
	char           *s;

	f = FTH_FLOAT_OBJECT(self);
	fs = fth_make_string_format("%s: ", FTH_INSTANCE_NAME(self));
	s = format_double(numbers_scratch, sizeof(numbers_scratch), f);
	return (fth_string_sformat(fs, "%s", s));
}

static FTH
fl_to_string(FTH self)
{
	ficlFloat 	f;
	char           *s;

	f = FTH_FLOAT_OBJECT(self);
	s = format_double(numbers_scratch, sizeof(numbers_scratch), f);
	return (fth_make_string(s));
}

static FTH
fl_copy(FTH self)
{
	return (fth_make_float(FTH_FLOAT_OBJECT(self)));
}

static FTH
fl_equal_p(FTH self, FTH obj)
{
	return (BOOL_TO_FTH(FTH_FLOAT_OBJECT(self) == FTH_FLOAT_OBJECT(obj)));
}

/*
 * Return a FTH float object from F.
 */
FTH
fth_make_float(ficlFloat f)
{
	FTH 		self;

	self = fth_make_instance(float_tag, NULL);
	FTH_FLOAT_OBJECT_SET(self, f);
	return (self);
}

FTH
fth_float_copy(FTH obj)
{
	if (FTH_FLOAT_T_P(obj))
		return (fl_copy(obj));
	return (obj);
}

static void
ficl_float_p(ficlVm *vm)
{
#define h_float_p "( obj -- f )  test if OBJ is a float number\n\
nil float? => #f\n\
1   float? => #f\n\
1.0 float? => #t\n\
Return #t if OBJ is a float object, otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_FLOAT_T_P(obj));
}

static void
ficl_inf_p(ficlVm *vm)
{
#define h_inf_p "( obj -- f )  test if OBJ is Infinite\n\
nil inf? => #f\n\
0   inf? => #f\n\
inf inf? => #t\n\
Return #t if OBJ is Infinite, otherwise #f.\n\
See also nan?, inf, nan."
	FTH 		obj;
	int 		flag;

	flag = 0;
	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);

	if (FTH_NUMBER_P(obj))
		flag = fth_isinf(fth_float_ref(obj));

	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_nan_p(ficlVm *vm)
{
#define h_nan_p "( obj -- f )  test if OBJ is Not a Number\n\
nil nan? => #f\n\
0   nan? => #f\n\
nan nan? => #t\n\
Return #t if OBJ is Not a Number, otherwise #f.\n\
See also inf?, inf, nan."
	FTH 		obj;
	int 		flag;

	flag = 0;
	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);

	if (FTH_NUMBER_P(obj))
		flag = fth_isnan(fth_float_ref(obj));

	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_inf(ficlVm *vm)
{
#define h_inf "( -- Inf )  Return Infinity."
	ficlStackPushFloat(vm->dataStack, FTH_INF);
}

static void
ficl_nan(ficlVm *vm)
{
#define h_nan "( -- NaN )  Return Not-A-Number."
	ficlStackPushFloat(vm->dataStack, FTH_NAN);
}

static void
ficl_f_dot_r(ficlVm *vm)
{
#define h_f_dot_r "( r n -- )  formatted number output\n\
17.0 3 f.r => |17.000 |\n\
17.0 6 f.r => |17.000000 |\n\
Print float R with N digits after decimal point."
	ficlFloat 	f;
	int 		n;

	FTH_STACK_CHECK(vm, 2, 0);
	n = (int) ficlStackPopInteger(vm->dataStack);
	f = ficlStackPopFloat(vm->dataStack);
	fth_printf("%.*f ", n, f);
}

static void
ficl_uf_dot_r(ficlVm *vm)
{
#define h_uf_dot_r "( r len-all len-after-comma -- )  formatted number\n\
17.0 8 3 uf.r => | 17.000 |\n\
17.0 8 2 uf.r => |  17.00 |\n\
Print float R in a right-adjusted field of LEN-ALL characters \
with LEN-AFTER-COMMA digits."
	ficlFloat 	f;
	int 		n, all;

	FTH_STACK_CHECK(vm, 2, 1);
	n = (int) ficlStackPopInteger(vm->dataStack);
	all = (int) ficlStackPopInteger(vm->dataStack);
	f = ficlStackPopFloat(vm->dataStack);
	fth_printf("%*.*f ", all, n, f);
}

static void
ficl_dfloats(ficlVm *vm)
{
#define h_dfloats "( n1 -- n2 )  return address units\n\
1 dfloats => 8\n\
4 dfloats => 32\n\
N2 is the number of address units of N1 dfloats (double)."
	ficlInteger 	n, s;

	FTH_STACK_CHECK(vm, 1, 1);
	n = ficlStackPopInteger(vm->dataStack);
	s = (ficlInteger) sizeof(ficlFloat);
	ficlStackPushInteger(vm->dataStack, n * s);
}

static void
ficl_falign(ficlVm *vm)
{
#define h_falign "( -- )  align dictionary"
	ficlDictionaryAlign(ficlVmGetDictionary(vm));
}

/*
 * Thanks to Sanjay Jain, we use expm1() and log1p()!
 */
static void
ficl_fexpm1(ficlVm *vm)
{
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 1);
	f = ficlStackPopFloat(vm->dataStack);
#if defined(HAVE_EXPM1)
#define h_fexpm1 "( x -- y )  y = expm1(x)"
	ficlStackPushFloat(vm->dataStack, expm1(f));
#else
#define h_fexpm1 "( x -- y )  y = exp(x) - 1.0"
	ficlStackPushFloat(vm->dataStack, exp(f) - 1.0);
#endif
}

static void
ficl_flogp1(ficlVm *vm)
{
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 1);
	f = ficlStackPopFloat(vm->dataStack);

	if (f >= 0.0) {
#if defined(HAVE_LOG1P)
#define h_flogp1 "( x -- y )  y = log1p(x)"
		ficlStackPushFloat(vm->dataStack, log1p(f));
#else
#define h_flogp1 "( x -- y )  y = log(x + 1.0)"
		ficlStackPushFloat(vm->dataStack, log(f + 1.0));
#endif
		return;
	}
	FTH_MATH_ERROR_THROW("log1p, x < 0");
	/* NOTREACHED */
}

static void
ficl_falog(ficlVm *vm)
{
#define h_falog "( x -- y )  y = pow(10.0, x)"
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 1);
	f = ficlStackPopFloat(vm->dataStack);
	ficlStackPushFloat(vm->dataStack, FTH_POW(10.0, f));
}

static void
ficl_fsincos(ficlVm *vm)
{
#define h_fsincos "( x -- y z )  y = sin(x), z = cos(x)"
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 2);
	f = ficlStackPopFloat(vm->dataStack);
	ficlStackPushFloat(vm->dataStack, sin(f));
	ficlStackPushFloat(vm->dataStack, cos(f));
}

N_FUNC_ONE_ARG(fabs, fabs, Float);
N_FUNC_ONE_ARG(floor, fth_floor, Float);
N_FUNC_ONE_ARG(fceil, fth_ceil, Float);
N_FUNC_ONE_ARG(ftrunc, fth_trunc, Float);
N_FUNC_ONE_ARG(fround, fth_rint, Float);
N_FUNC_ONE_ARG(fsqrt, sqrt, Float);
N_FUNC_ONE_ARG(fexp, exp, Float);
N_FUNC_ONE_ARG(flog, fth_log, Float);
N_FUNC_ONE_ARG(flog2, fth_log2, Float);
N_FUNC_ONE_ARG(flog10, fth_log10, Float);
N_FUNC_ONE_ARG(fsin, sin, Float);
N_FUNC_ONE_ARG(fcos, cos, Float);
N_FUNC_ONE_ARG(ftan, tan, Float);
N_FUNC_ONE_ARG(fasin, asin, Float);
N_FUNC_ONE_ARG(facos, acos, Float);
N_FUNC_ONE_ARG(fatan, atan, Float);
N_FUNC_ONE_ARG(fsinh, sinh, Float);
N_FUNC_ONE_ARG(fcosh, cosh, Float);
N_FUNC_ONE_ARG(ftanh, tanh, Float);
N_FUNC_ONE_ARG(fasinh, asinh, Float);
N_FUNC_ONE_ARG(facosh, acosh, Float);
N_FUNC_ONE_ARG(fatanh, atanh, Float);
N_FUNC_TWO_ARGS(fmod, fmod, Float);
N_FUNC_TWO_ARGS(fpow, fth_pow, Float);
N_FUNC_TWO_ARGS(fatan2, atan2, Float);

/*
 * Parse ficlInteger, ficl2Integer, ficlUnsigned, ficl2Unsigned, and
 * ficlFloat (1, 1., 1.0, 1e, etc).
 */
int
ficl_parse_number(ficlVm *vm, ficlString s)
{
	int 		base;
	char           *test;
	char           *str;
	ficlInteger 	i;
	ficlUnsigned 	u;
	ficl2Integer 	di;
	ficl2Unsigned 	ud;
	ficlFloat 	f;

	if (s.length < 1 || s.length >= FICL_PAD_SIZE)
		return (FICL_FALSE);

	base = (int) vm->base;
	str = vm->pad;
	strncpy(str, s.text, s.length);
	str[s.length] = '\0';

	/* ficlInteger */
	i = strtol(str, &test, base);

	if (*test == '\0' && errno != ERANGE) {
		ficlStackPushInteger(vm->dataStack, i);
		goto okay;
	}
	/* 3e => 3. */
	if (str[s.length - 1] == 'e')
		str[s.length - 1] = '.';

	/* ficlFloat */
	f = strtod(str, &test);

	if (*test == '\0' && errno != ERANGE) {
		ficlStackPushFloat(vm->dataStack, f);
		goto okay;
	}
	/* ficl2Integer */
	di = strtoll(str, &test, base);

	if (*test == '\0' && errno != ERANGE) {
		ficlStackPush2Integer(vm->dataStack, di);
		goto okay;
	}
	/* ficlUnsigned */
	u = strtoul(str, &test, base);

	if (*test == '\0' && errno != ERANGE) {
		ficlStackPushUnsigned(vm->dataStack, u);
		goto okay;
	}
	/* ficl2Unsigned */
	ud = strtoull(str, &test, base);

	if (*test == '\0' && errno != ERANGE) {
		ficlStackPush2Unsigned(vm->dataStack, ud);
		goto okay;
	}
	errno = 0;
	return (FICL_FALSE);

okay:
	errno = 0;

	if (vm->state == FICL_VM_STATE_COMPILE)
		ficlPrimitiveLiteralIm(vm);

	return (FICL_TRUE);
}

/* === COMPLEX === */

#if HAVE_COMPLEX

#define h_list_of_complex_functions "\
*** COMPLEX PRIMITIVES ***\n\
complex?  real-ref  imag-ref (image-ref)\n\
make-rectangular (>complex)\n\
make-polar\n\
c.   s>c  c>s  f>c  c>f  q>c  (r>c)  >c\n\
c0=  c0<> c=   c<>\n\
c+   c-   c*   c/   1/c\n\
carg      cabs (magnitude)    cabs2\n\
c**  (cpow)    conj (conjugate)\n\
csqrt     cexp      clog      clog10\n\
csin      ccos      ctan\n\
casin     cacos     catan     catan2\n\
csinh     ccosh     ctanh\n\
casinh    cacosh    catanh\n\
See also long-long and float."

static char 	numbers_scratch_02[BUFSIZ];

static FTH
cp_inspect(FTH self)
{
	char           *re;
	char           *im;
	size_t 		size;
	FTH		fs;

	re = numbers_scratch;
	im = numbers_scratch_02;
	size = sizeof(numbers_scratch);
	fs = fth_make_string_format("%s: ", FTH_INSTANCE_NAME(self));
	fth_string_scat(fs, "real ");
	fth_string_scat(fs, format_double(re, size, FTH_COMPLEX_REAL(self)));
	fth_string_scat(fs, ", image ");
	fth_string_scat(fs, format_double(im, size, FTH_COMPLEX_IMAG(self)));
	return (fs);
}

static FTH
cp_to_string(FTH self)
{
	char           *re;
	char           *im;
	size_t 		size;
	FTH		fs;

	re = numbers_scratch;
	im = numbers_scratch_02;
	size = sizeof(numbers_scratch);
	fs = fth_make_string(format_double(re, size, FTH_COMPLEX_REAL(self)));
	format_double(im, size, FTH_COMPLEX_IMAG(self));

	if (im[0] != '+' && im[0] != '-')
		fth_string_scat(fs, "+");

	fth_string_scat(fs, im);
	fth_string_scat(fs, "i");
	return (fs);
}

static FTH
cp_copy(FTH self)
{
	return (fth_make_complex(FTH_COMPLEX_OBJECT(self)));
}

static FTH
cp_equal_p(FTH self, FTH obj)
{
	return (BOOL_TO_FTH(FTH_COMPLEX_REAL(self) == FTH_COMPLEX_REAL(obj) &&
		FTH_COMPLEX_IMAG(self) == FTH_COMPLEX_IMAG(obj)));
}

#endif				/* HAVE_COMPLEX */

static void
ficl_complex_p(ficlVm *vm)
{
#define h_complex_p "( obj -- f )  test if OBJ is a complex number\n\
nil complex? => #f\n\
1   complex? => #f\n\
1+i complex? => #t\n\
Return #t if OBJ is a complex object, otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_COMPLEX_P(obj));
}

static void
ficl_creal(ficlVm *vm)
{
#define h_creal "( numb -- re )  return number's real part\n\
1   real-ref => 1.0\n\
1.0 real-ref => 1.0\n\
1+i real-ref => 1.0\n\
Return the real part of NUMB.\n\
See also imag-ref."
	ficlFloat 	f;
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
#if HAVE_COMPLEX
	if (FTH_COMPLEX_P(obj))
		f = FTH_COMPLEX_REAL(obj);
	else
		f = fth_float_ref(obj);
#else
	f = fth_float_ref(obj);
#endif
	ficlStackPushFloat(vm->dataStack, f);
}

static void
ficl_cimage(ficlVm *vm)
{
#define h_cimage "( numb -- im )  return number's image part\n\
1   imag-ref => 0.0\n\
1.0 imag-ref => 0.0\n\
1+i imag-ref => 1.0\n\
Return the image part of NUMB.\n\
See also real-ref."
	FTH 		obj;
	ficlFloat 	f;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	f = 0.0;
#if HAVE_COMPLEX
	if (FTH_COMPLEX_P(obj))
		f = FTH_COMPLEX_IMAG(obj);
#endif
	ficlStackPushFloat(vm->dataStack, f);
}

#if HAVE_COMPLEX

/*
 * Return a FTH complex object from Z.
 */
FTH
fth_make_complex(ficlComplex z)
{
	FTH 		self;

	self = fth_make_instance(complex_tag, NULL);
	FTH_COMPLEX_OBJECT_SET(self, z);
	return (self);
}

FTH
fth_make_rectangular(ficlFloat real, ficlFloat image)
{
	return (fth_make_complex(real + image * _Complex_I));
}

static void
ficl_complex_i(ficlVm *vm)
{
#define h_complex_i "( -- I )  return _Complex_I\n\
1 Complex-I c* => 0.0+1.0i\n\
1i => 0.0+1.0i\n\
0+1i => 0.0+1.0i\n\
-1 Complex-I c* value -z1\n\
-z1 => -0.0-1.0i\n\
3+i value z3\n\
z3 => 3.0+1.0i\n\
z3 -z1 c* => 1.0-3.0i"
	FTH_STACK_CHECK(vm, 0, 1);
	ficlStackPushComplex(vm->dataStack, _Complex_I);
}

static void
ficl_cnegate(ficlVm *vm)
{
#define h_cnegate "( z -- -z )  z * (-1.0 * _Complex_I)\n\
3+i => 3.0+1.0i\n\
3+1 cnegate => 1.0-3.0i"
	ficlComplex	cp;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);
	ficlStackPushComplex(vm->dataStack, cp * (-1.0 * _Complex_I));
}

static void
ficl_make_complex_rectangular(ficlVm *vm)
{
#define h_make_complex_rectangular "( real image -- complex )  complex numb\n\
1 1 make-rectangular => 1.0+1.0i\n\
Return complex object with REAL and IMAGE part.\n\
See also make-polar."
	ficlFloat 	real;
	ficlFloat 	image;

	FTH_STACK_CHECK(vm, 2, 1);
	image = fth_float_ref(fth_pop_ficl_cell(vm));
	real = fth_float_ref(fth_pop_ficl_cell(vm));
	ficlStackPushFTH(vm->dataStack, fth_make_rectangular(real, image));
}

static ficlComplex
make_polar(ficlFloat real, ficlFloat theta)
{
	return (real * cos(theta) + real * sin(theta) * _Complex_I);
}

FTH
fth_make_polar(ficlFloat real, ficlFloat theta)
{
	return (fth_make_complex(make_polar(real, theta)));
}

static void
ficl_make_complex_polar(ficlVm *vm)
{
#define h_make_complex_polar "( real theta -- complex )  polar complex numb\n\
1 1 make-polar => 0.540302+0.841471i\n\
Return polar complex object from REAL and THETA.\n\
See also make-rectangular."
	ficlFloat 	real;
	ficlFloat 	theta;

	FTH_STACK_CHECK(vm, 2, 1);
	theta = fth_float_ref(fth_pop_ficl_cell(vm));
	real = fth_float_ref(fth_pop_ficl_cell(vm));
	ficlStackPushFTH(vm->dataStack, fth_make_polar(real, theta));
}

static void
ficl_c_dot(ficlVm *vm)
{
#define h_c_dot "( c -- )  print number\n\
1+i c. => |1.0+1.0i |\n\
Print complex number C."
	FTH_STACK_CHECK(vm, 1, 0);
	fth_printf("%S ", cp_to_string(fth_pop_ficl_cell(vm)));
}

static void
ficl_creciprocal(ficlVm *vm)
{
#define h_creciprocal "( x -- y )  y = 1 / x"
	ficlComplex 	cp;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);
	ficlStackPushComplex(vm->dataStack, 1.0 / cp);
}

static void
ficl_ceqz(ficlVm *vm)
{
#define h_ceqz "( x -- f )  x == 0 => flag"
	ficlComplex 	cp;
	int		flag;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);
	flag = FICL_TRUE;

	if (creal(cp) != 0.0 || cimag(cp) != 0.0)
		flag = FICL_FALSE;

	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_cnoteqz(ficlVm *vm)
{
#define h_cnoteqz "( x -- f )  x != 0 => flag"
	ficlComplex 	cp;
	int		flag;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);
	flag = FICL_FALSE;

	if (creal(cp) != 0.0 || cimag(cp) != 0.0)
		flag = FICL_TRUE;

	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_ceq(ficlVm *vm)
{
#define h_ceq "( x y -- f )  x == y => flag"
	ficlComplex 	x, y;
	int		flag;

	FTH_STACK_CHECK(vm, 2, 1);
	y = ficlStackPopComplex(vm->dataStack);
	x = ficlStackPopComplex(vm->dataStack);
	flag = FICL_TRUE;

	if (creal(x) != creal(y) || cimag(x) != cimag(y))
		flag = FICL_FALSE;

	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_cnoteq(ficlVm *vm)
{
#define h_cnoteq "( x y -- f )  x != y => flag"
	ficlComplex 	x, y;
	int		flag;

	FTH_STACK_CHECK(vm, 2, 1);
	y = ficlStackPopComplex(vm->dataStack);
	x = ficlStackPopComplex(vm->dataStack);
	flag = FICL_FALSE;

	if (creal(x) != creal(y) || cimag(x) != cimag(y))
		flag = FICL_TRUE;

	ficlStackPushBoolean(vm->dataStack, flag);
}

N_FUNC_TWO_ARGS_OP(cadd, +, Complex);
N_FUNC_TWO_ARGS_OP(csub, -, Complex);
N_FUNC_TWO_ARGS_OP(cmul, *, Complex);
N_FUNC_TWO_ARGS_OP(cdiv, /, Complex);

N_FUNC_ONE_ARG(carg, carg, Complex);
N_FUNC_ONE_ARG(cabs, cabs, Complex);
N_FUNC_ONE_ARG(cabs2, cabs2, Complex);
N_FUNC_TWO_ARGS(cpow, cpow, Complex);
N_FUNC_ONE_ARG(cconj, conj, Complex);
N_FUNC_ONE_ARG(csqrt, csqrt, Complex);
N_FUNC_ONE_ARG(cexp, cexp, Complex);
N_FUNC_ONE_ARG(clog, clog, Complex);
N_FUNC_ONE_ARG(clog10, clog10, Complex);
N_FUNC_ONE_ARG(csin, csin, Complex);
N_FUNC_ONE_ARG(ccos, ccos, Complex);
N_FUNC_ONE_ARG(ctan, ctan, Complex);
N_FUNC_ONE_ARG(catan, catan, Complex);
N_FUNC_TWO_ARGS(catan2, catan2, Complex);
N_FUNC_ONE_ARG(csinh, csinh, Complex);
N_FUNC_ONE_ARG(ccosh, ccosh, Complex);
N_FUNC_ONE_ARG(ctanh, ctanh, Complex);
N_FUNC_ONE_ARG(casinh, casinh, Complex);

static void
ficl_casin(ficlVm *vm)
{
#define h_casin "( x -- y ) y = casin(x)"
	ficlComplex	cp;
	ficlComplex	z;
	ficlFloat	f;
	ficlFloat	r;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);

	/* complex */
	if (cimag(cp) != 0.0) {
		ficlStackPushComplex(vm->dataStack, casin(cp));
		return;
	}

	/* float */
	f = fabs(creal(cp));
	r = 1.0 / f;

	if (f <= 1.0) {
		ficlStackPushFloat(vm->dataStack, asin(creal(cp)));
		return;
	} else {
		z = M_PI_2 - (_Complex_I *
		    clog(f * (1.0 + (sqrt(1.0 + r) * csqrt(1.0 - r)))));

		if (creal(cp) < 0.0)
			z = -z;

		ficlStackPushComplex(vm->dataStack, z);
	}

}

static void
ficl_cacos(ficlVm *vm)
{
#define h_cacos "( x -- y ) y = cacos(x)"
	ficlComplex	cp;
	ficlComplex	z;
	ficlFloat	f;
	ficlFloat	r;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);

	/* complex */
	if (cimag(cp) != 0.0) {
		ficlStackPushComplex(vm->dataStack, cacos(cp));
		return;
	}

	/* float */
	f = fabs(creal(cp));
	r = 1.0 / f;

	if (f <= 1.0) {
		ficlStackPushFloat(vm->dataStack, acos(creal(cp)));
		return;
	} else {
		z = _Complex_I *
		    clog(f * (1.0 + (sqrt(1.0 + r) * csqrt(1.0 - r))));

		if (creal(cp) <= 0.0)
			z = M_PI - z;

		ficlStackPushComplex(vm->dataStack, z);
	}
}

static void
ficl_cacosh(ficlVm *vm)
{
#define h_cacosh "( x -- y ) y = cacosh(x)"
	ficlComplex	cp;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);

	/* complex */
	if (cimag(cp) != 0.0 || creal(cp) < 1.0)
		ficlStackPushComplex(vm->dataStack, cacosh(cp));
	else
		ficlStackPushFloat(vm->dataStack, acosh(creal(cp)));
}

static void
ficl_catanh(ficlVm *vm)
{
#define h_catanh "( x -- y ) y = catanh(x)"
	ficlComplex	cp;

	FTH_STACK_CHECK(vm, 1, 1);
	cp = ficlStackPopComplex(vm->dataStack);

	/* complex */
	if (cimag(cp) != 0.0 || fabs(creal(cp)) >= 1.0)
		ficlStackPushComplex(vm->dataStack, catanh(cp));
	else
		ficlStackPushFloat(vm->dataStack, atanh(creal(cp)));
}

/*
 * Parse ficlComplex (1i, 1-i, -1+1i, 1.0+1.0i, etc).
 *	1i	==> 0.0+1.0i
 *      1+i	==> 1.0+1.0i
 *      1-i	==> 1.0-1.0i
 */
int
ficl_parse_complex(ficlVm *vm, ficlString s)
{
	ficlFloat 	re;
	ficlFloat 	im;
	size_t 		loc_len;
	char           *locp;
	char           *locn;
	char           *test;
	char           *loc;
	char           *sreal;
	char           *simag;
	char 		re_buf[FICL_PAD_SIZE];

	if (s.length < 2 || tolower((int) s.text[s.length - 1]) != 'i')
		return (FICL_FALSE);

	if (s.length >= FICL_PAD_SIZE)
		return (FICL_FALSE);

	re = 0.0;
	im = 0.0;
	sreal = re_buf;
	simag = vm->pad;
	strncpy(simag, s.text, s.length);
	simag[s.length] = '\0';
	locp = strrchr(simag, '+');
	locn = strrchr(simag, '-');
	loc = FICL_MAX(locp, locn);

	if (loc == NULL) {
		loc = strrchr(simag, 'i');

		if (loc == NULL)
			loc = strrchr(simag, 'I');
	}

	if (loc == NULL)
		return (FICL_FALSE);

	strncpy(sreal, simag, (size_t) (loc - simag));
	sreal[loc - simag] = '\0';
	re = strtod(sreal, &test);

	if (*test != '\0' || errno == ERANGE) {
		errno = 0;
		return (FICL_FALSE);
	}

	loc_len = fth_strlen(loc);	/* skip \0 above */

	if (loc_len > 2) {
		loc[loc_len - 1] = '\0';	/* discard trailing i */
		im = strtod(loc, &test);
		if (*test != '\0' || errno == ERANGE)
			return (FICL_FALSE);
	} else {
		switch(loc[0]) {
		case '+':
			im = 1.0;
			break;
		case '-':
			im = -1.0;
			break;
		case 'I':
		case 'i':
			/*
			 * changed on Sat Dec 28 14:17:46 CET 2019
			 * before: 3i ==> 3.0+1.0i
			 *    now: 3i ==> 0.0+3.0i
			 */
			im = re;
			re = 0.0;
			break;
		default:
			return (FICL_FALSE);
			break;
		}
	}
	ficlStackPushFTH(vm->dataStack, fth_make_rectangular(re, im));

	if (vm->state == FICL_VM_STATE_COMPILE)
		ficlPrimitiveLiteralIm(vm);

	return (FICL_TRUE);
}

#endif				/* HAVE_COMPLEX */

/* === BIGNUM via xedit/lisp/mp === */

#define h_list_of_bignum_functions "\
*** BIGNUMB PRIMITIVES ***\n\
bignum?   >bignum   bn.\n\
s>b  b>s  f>b  b>f\n\
b0=  b0<> b0<  b0>  b0<= b0>=\n\
b=   b<>  b<   b>   b<=  b>=\n\
b+   b-   b*   b/\n\
bgcd blcm b** (bpow)\n\
broot bsqrt\n\
bnegate   babs bmin bmax\n\
b2*  b2/  bmod b/mod blshift brshift"

static void
ficl_bignum_p(ficlVm *vm)
{
#define h_bignum_p "( obj -- f )  test if OBJ is a bignum\n\
nil bignum? => #f\n\
1e100 bignum? => #f\n\
12345678901234567890 bignum? => #t\n\
Return #t if OBJ is a bignum object, otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_BIGNUM_P(obj));
}

static FTH
bn_inspect(FTH self)
{
	return (fth_make_string_format("%s: %S",
		FTH_INSTANCE_NAME(self), bn_to_string(self)));
}

static FTH
bn_to_string(FTH self)
{
	FTH 		fs;
	char           *buf;

	buf = mpi_getstr(NULL, FTH_BIGNUM_OBJECT(self), 10);
	fs = fth_make_string(buf);
	mp_free(buf);
	return (fs);
}

static FTH
bn_copy(FTH self)
{
	ficlBignum 	res;

	res = mpi_new();
	mpi_set(res, FTH_BIGNUM_OBJECT(self));
	return (fth_make_bignum(res));
}

static FTH
bn_equal_p(FTH self, FTH obj)
{
	int		flag;

	flag = mpi_cmp(FTH_BIGNUM_OBJECT(self), FTH_BIGNUM_OBJECT(obj));
	return (BOOL_TO_FTH(flag == 0));
}

static void
bn_free(FTH self)
{
	mpi_free(FTH_BIGNUM_OBJECT(self));
}

enum {
	BN_ADD,
	BN_SUB,
	BN_MUL,
	BN_DIV
};

static ficlBignum
bn_math(FTH m, FTH n, int type)
{
	ficlBignum 	x;
	ficlBignum 	y;
	ficlBignum 	z;

	x = fth_bignum_ref(m);
	y = fth_bignum_ref(n);
	z = mpi_new();

	switch (type) {
	case BN_ADD:
		mpi_add(z, x, y);
		break;
	case BN_SUB:
		mpi_sub(z, x, y);
		break;
	case BN_MUL:
		mpi_mul(z, x, y);
		break;
	case BN_DIV:
	default:
		mpi_div(z, x, y);
		break;
	}

	mpi_free(x);
	mpi_free(y);
	return (z);
}

static FTH
bn_add(FTH m, FTH n)
{
	return (fth_make_bignum(bn_math(m, n, BN_ADD)));
}

static FTH
bn_sub(FTH m, FTH n)
{
	return (fth_make_bignum(bn_math(m, n, BN_SUB)));
}

static FTH
bn_mul(FTH m, FTH n)
{
	return (fth_make_bignum(bn_math(m, n, BN_MUL)));
}

static FTH
bn_div(FTH m, FTH n)
{
	return (fth_make_bignum(bn_math(m, n, BN_DIV)));
}

FTH
fth_make_bignum(ficlBignum m)
{
	FTH 		self;

	self = fth_make_instance(bignum_tag, NULL);
	FTH_BIGNUM_OBJECT_SET(self, m);
	return (self);
}

static ficlBignum
mpi_new(void)
{
	ficlBignum	bn;

	bn = mp_malloc(sizeof(mpi));
	mpi_init(bn);
	return (bn);
}

static void
mpi_free(ficlBignum bn)
{
	mpi_clear(bn);
	mp_free(bn);
}

FTH
fth_make_big(FTH m)
{
	return (fth_make_bignum(fth_bignum_ref(m)));
}

static void
ficl_bn_dot(ficlVm *vm)
{
#define h_bn_dot "( numb -- )  number output\n\
1 >bignum bn. => 1\n\
Print bignum number NUMB with space added."
	ficlBignum	x;
	char           *str;

	FTH_STACK_CHECK(vm, 1, 0);
	x = ficlStackPopBignum(vm->dataStack);
	str = mpi_getstr(NULL, x, 10);
	fth_printf("%s ", str);
	mp_free(str);
	mpi_free(x);
}

#define N_BIGNUM_FUNC_TEST_ZERO(Name, OP)				\
static int								\
fth_bn_ ## Name(FTH m)							\
{									\
	int		flag;						\
									\
	if (FTH_BIGNUM_P(m))						\
		flag = (mpi_cmpi(FTH_BIGNUM_OBJECT(m), 0) OP 0);	\
	else {								\
		ficlBignum	x;					\
									\
		x = fth_bignum_ref(m);					\
		flag = (mpi_cmpi(x, 0) OP 0);				\
		mpi_free(x);						\
	}								\
	return (flag);							\
}									\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	int		flag;						\
									\
	FTH_STACK_CHECK(vm, 1, 1);					\
	flag = fth_bn_ ## Name(fth_pop_ficl_cell(vm));			\
	ficlStackPushBoolean(vm->dataStack, flag);			\
}									\
static char* h_ ## Name = "( x -- f )  x " #OP " 0 => flag (bignum)"

/*-
 * build:
 *   int  fth_bn_beqz(FTH m) ...    for C fth_number_equal_p etc
 *   void ficl_beqz(ficlVm *vm) ... for Forth words
 */
N_BIGNUM_FUNC_TEST_ZERO(beqz, ==);
N_BIGNUM_FUNC_TEST_ZERO(bnoteqz, !=);
N_BIGNUM_FUNC_TEST_ZERO(blessz, <);
N_BIGNUM_FUNC_TEST_ZERO(bgreaterz, >);
N_BIGNUM_FUNC_TEST_ZERO(blesseqz, <=);
N_BIGNUM_FUNC_TEST_ZERO(bgreatereqz, >=);

#define N_BIGNUM_FUNC_TEST_TWO_OP(Name, OP)				\
static int								\
fth_bn_ ## Name(FTH m, FTH n)						\
{									\
	int		flag;						\
									\
	if (FTH_BIGNUM_P(m)) {						\
		if (FTH_BIGNUM_P(n))					\
			flag = (mpi_cmp(FTH_BIGNUM_OBJECT(m),		\
			    FTH_BIGNUM_OBJECT(n)) OP 0);		\
		else {							\
			ficlBignum	y; 				\
									\
			y = fth_bignum_ref(n);				\
			flag = (mpi_cmp(FTH_BIGNUM_OBJECT(m), y) OP 0);	\
			mpi_free(y);					\
		}							\
	} else if (FTH_BIGNUM_P(n)) {					\
		ficlBignum	x;					\
									\
		x = fth_bignum_ref(m);					\
		flag = (mpi_cmp(x, FTH_BIGNUM_OBJECT(n)) OP 0);		\
		mpi_free(x);						\
	} else {							\
		ficlBignum	x;					\
		ficlBignum	y;					\
									\
		x = fth_bignum_ref(m);					\
		y = fth_bignum_ref(n);					\
		flag = (mpi_cmp(x, y) OP 0);				\
		mpi_free(x);						\
		mpi_free(y);						\
	}								\
	return (flag);							\
}									\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	FTH		m;						\
	FTH		n;						\
	int		flag;						\
									\
	FTH_STACK_CHECK(vm, 2, 1);					\
	n = fth_pop_ficl_cell(vm);					\
	m = fth_pop_ficl_cell(vm);					\
	flag = fth_bn_ ## Name(m, n);					\
	ficlStackPushBoolean(vm->dataStack, flag);			\
}									\
static char* h_ ## Name = "( x y -- f )  x " #OP " y => flag (bignum)"

/*-
 * build: 
 *   int  fth_bn_beq(FTH m, FTH n) ... for C fth_number_equal_p etc
 *   void ficl_beq(ficlVm *vm) ...     for Forth words
 */
N_BIGNUM_FUNC_TEST_TWO_OP(beq, ==);
N_BIGNUM_FUNC_TEST_TWO_OP(bnoteq, !=);
N_BIGNUM_FUNC_TEST_TWO_OP(bless, <);
N_BIGNUM_FUNC_TEST_TWO_OP(bgreater, >);
N_BIGNUM_FUNC_TEST_TWO_OP(blesseq, <=);
N_BIGNUM_FUNC_TEST_TWO_OP(bgreatereq, >=);

#define N_BIGNUM_MATH_FUNC_OP(Name, OP, FName)				\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	FTH		m;						\
	FTH		n;						\
									\
	FTH_STACK_CHECK(vm, 2, 1);					\
	n = fth_pop_ficl_cell(vm);					\
	m = fth_pop_ficl_cell(vm);					\
	ficlStackPushFTH(vm->dataStack, FName(m, n));			\
}									\
static char* h_ ## Name = "( x y -- z )  z = x " #OP " y (bignum)"

N_BIGNUM_MATH_FUNC_OP(badd, +, bn_add);
N_BIGNUM_MATH_FUNC_OP(bsub, -, bn_sub);
N_BIGNUM_MATH_FUNC_OP(bmul, *, bn_mul);
N_BIGNUM_MATH_FUNC_OP(bdiv, /, bn_div);

static void
ficl_bgcd(ficlVm *vm)
{
#define h_bgcd "( x y -- z )  z = gcd(x, y)"
	ficlBignum 	x;
	ficlBignum 	y;
	ficlBignum 	z;

	FTH_STACK_CHECK(vm, 2, 1);
	z = mpi_new();
	y = ficlStackPopBignum(vm->dataStack);
	x = ficlStackPopBignum(vm->dataStack);
	mpi_gcd(z, x, y);
	mpi_free(x);
	mpi_free(y);
	ficlStackPushBignum(vm->dataStack, z);
}

static void
ficl_blcm(ficlVm *vm)
{
#define h_blcm "( x y -- z )  z = lcm(x, y)"
	ficlBignum 	x;
	ficlBignum 	y;
	ficlBignum 	z;

	FTH_STACK_CHECK(vm, 2, 1);
	z = mpi_new();
	y = ficlStackPopBignum(vm->dataStack);
	x = ficlStackPopBignum(vm->dataStack);
	mpi_lcm(z, x, y);
	mpi_free(x);
	mpi_free(y);
	ficlStackPushBignum(vm->dataStack, z);
}

static void
ficl_bpow(ficlVm *vm)
{
#define h_bpow "( x y -- z )  z = x ** y"
	ficlBignum 	x;
	ficlUnsigned	y;
	ficlBignum 	z;

	FTH_STACK_CHECK(vm, 2, 1);
	z = mpi_new();
	y = ficlStackPopUnsigned(vm->dataStack);
	x = ficlStackPopBignum(vm->dataStack);
	mpi_pow(z, x, y);
	mpi_free(x);
	ficlStackPushBignum(vm->dataStack, z);
}

static void
ficl_broot(ficlVm *vm)
{
#define h_broot "( b1 u -- b2 n )  b2 = root(b1, uth);  \
n=1 if exact, n=0 otherwise"
	ficlBignum 	b1;
	ficlUnsigned	u;
	ficlBignum 	b2;
	ficlInteger	n;

	FTH_STACK_CHECK(vm, 2, 2);
	u = ficlStackPopUnsigned(vm->dataStack);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	n = mpi_root(b2, b1, u);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
	ficlStackPushInteger(vm->dataStack, n);
}

static void
ficl_bsqrt(ficlVm *vm)
{
#define h_bsqrt "( b1 -- b2 n )  b2 = sqrt(b1);  n=1 if exact, n=0 otherwise"
	ficlBignum 	b1;
	ficlBignum 	b2;
	ficlInteger	n;

	FTH_STACK_CHECK(vm, 1, 2);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	n = mpi_sqrt(b2, b1);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
	ficlStackPushInteger(vm->dataStack, n);
}

static void
ficl_bnegate(ficlVm *vm)
{
	ficlBignum 	b1;
	ficlBignum 	b2;

	FTH_STACK_CHECK(vm, 1, 1);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	mpi_neg(b2, b1);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
}

static void
ficl_babs(ficlVm *vm)
{
	ficlBignum 	b1;
	ficlBignum 	b2;

	FTH_STACK_CHECK(vm, 1, 1);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	mpi_abs(b2, b1);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
}

static void
ficl_bmin(ficlVm *vm)
{
	ficlBignum 	x;
	ficlBignum 	y;

	FTH_STACK_CHECK(vm, 2, 1);
	y = ficlStackPopBignum(vm->dataStack);
	x = ficlStackPopBignum(vm->dataStack);

	if (mpi_cmp(x, y) < 0) {
		mpi_free(y);
		ficlStackPushBignum(vm->dataStack, x);
	} else {
		mpi_free(x);
		ficlStackPushBignum(vm->dataStack, y);
	}
}

static void
ficl_bmax(ficlVm *vm)
{
	ficlBignum 	x;
	ficlBignum 	y;

	FTH_STACK_CHECK(vm, 2, 1);
	y = ficlStackPopBignum(vm->dataStack);
	x = ficlStackPopBignum(vm->dataStack);

	if (mpi_cmp(x, y) >= 0) {
		mpi_free(y);
		ficlStackPushBignum(vm->dataStack, x);
	} else {
		mpi_free(x);
		ficlStackPushBignum(vm->dataStack, y);
	}
}

static void
ficl_btwostar(ficlVm *vm)
{
	ficlBignum 	b1;
	ficlBignum 	b2;

	FTH_STACK_CHECK(vm, 1, 1);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	mpi_ash(b2, b1, 1);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
}

static void
ficl_btwoslash(ficlVm *vm)
{
	ficlBignum 	b1;
	ficlBignum 	b2;

	FTH_STACK_CHECK(vm, 1, 1);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	mpi_ash(b2, b1, -1);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
}

static void
ficl_bmod(ficlVm *vm)
{
#define h_bmod "( b1 b2 -- b3 )  b3 = b1 % b2"
	ficlBignum 	b1;
	ficlBignum 	b2;
	ficlBignum 	b3;

	FTH_STACK_CHECK(vm, 2, 1);
	b3 = mpi_new();
	b2 = ficlStackPopBignum(vm->dataStack);
	b1 = ficlStackPopBignum(vm->dataStack);
	mpi_mod(b3, b1, b2);
	mpi_free(b1);
	mpi_free(b2);
	ficlStackPushBignum(vm->dataStack, b3);
}

static void
ficl_bslashmod(ficlVm *vm)
{
#define h_bslashmod "( b1 b2 -- b3 b4 )  b1 / b2; b3 = remainder; b4 = quotient"
	ficlBignum 	b1;
	ficlBignum 	b2;
	ficlBignum 	b3;
	ficlBignum 	b4;

	FTH_STACK_CHECK(vm, 2, 2);
	b4 = mpi_new();
	b3 = mpi_new();
	b2 = ficlStackPopBignum(vm->dataStack);
	b1 = ficlStackPopBignum(vm->dataStack);
	mpi_divqr(b4, b3, b1, b2);
	mpi_free(b1);
	mpi_free(b2);
	ficlStackPushBignum(vm->dataStack, b3);
	ficlStackPushBignum(vm->dataStack, b4);
}

static void
ficl_blshift(ficlVm *vm)
{
#define h_blshift "( b1 n -- b2 )  b2 = b1 * 2^n"
	ficlBignum 	b1;
	ficlInteger	n;
	ficlBignum 	b2;

	FTH_STACK_CHECK(vm, 2, 1);
	n = ficlStackPopInteger(vm->dataStack);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	mpi_ash(b2, b1, n);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
}

static void
ficl_brshift(ficlVm *vm)
{
#define h_brshift "( b1 n -- b2 )  b2 = b1 / 2^n"
	ficlBignum 	b1;
	ficlInteger	n;
	ficlBignum 	b2;

	FTH_STACK_CHECK(vm, 2, 1);
	n = ficlStackPopInteger(vm->dataStack);
	b1 = ficlStackPopBignum(vm->dataStack);
	b2 = mpi_new();
	mpi_ash(b2, b1, -n);
	mpi_free(b1);
	ficlStackPushBignum(vm->dataStack, b2);
}

/*
 * Parse ficlBignum (in base 10) via xedit/lisp/mp.
 */
int
ficl_parse_bignum(ficlVm *vm, ficlString s)
{
	ficlBignum 	bn;

	if (s.length < 10)
		return (FICL_FALSE);
	
	bn = mpi_new();
	mpi_setstr(bn, s.text, 10);
	ficlStackPushBignum(vm->dataStack, bn);

	if (vm->state == FICL_VM_STATE_COMPILE)
		ficlPrimitiveLiteralIm(vm);

	return (FICL_TRUE);
}

/* === RATIO via xedit/lisp/mp === */

#define h_list_of_ratio_functions "\
*** RATIONAL PRIMITIVES ***\n\
ratio? (rational?)  make-ratio  >ratio\n\
q.   rationalize\n\
s>q  q>s  c>q  f>q  q>f\n\
q0=  q0<> q0<  q0>  q0<= q0>=\n\
q=   q<>  q<   q>   q<=  q>=\n\
q+   q-   q*   q/   1/q  q** (qpow)\n\
qnegate   qfloor    qceil    qabs\n\
and some aliases:\n\
r.   1/r  s>r  r>s  c>r  f>r  r>f\n\
rnegate   rfloor    rceil    rabs\n\
exact->inexact      inexact->exact\n\
numerator      denominator"

static void
ficl_ratio_p(ficlVm *vm)
{
#define h_ratio_p "( obj -- f )  test if OBJ is a rational number\n\
nil    ratio? => #f\n\
1/2    ratio? => #t\n\
pi f>r ratio? => #t\n\
Return #t if OBJ is a ratio object, otherwise #f."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 1);
	obj = fth_pop_ficl_cell(vm);
	ficlStackPushBoolean(vm->dataStack, FTH_RATIO_P(obj));
}

#define FTH_RATIO_NUM(Obj)	mpr_num(FTH_RATIO_OBJECT(Obj))
#define FTH_RATIO_DEN(Obj)	mpr_den(FTH_RATIO_OBJECT(Obj))

static FTH
rt_inspect(FTH self)
{
	return (fth_make_string_format("%s: %S",
		FTH_INSTANCE_NAME(self), rt_to_string(self)));
}

static FTH
rt_to_string(FTH self)
{
	FTH 		fs;
	char           *buf;

	buf = mpr_getstr(NULL, FTH_RATIO_OBJECT(self), 10);
	fs = fth_make_string(buf);
	mp_free(buf);
	return (fs);
}

static FTH
rt_copy(FTH self)
{
	ficlRatio 	res;

	res = mpr_new();
	mpr_set(res, FTH_RATIO_OBJECT(self));
	return (fth_make_rational(res));
}

static FTH
rt_equal_p(FTH self, FTH obj)
{
	int 		flag;

	flag = mpr_cmp(FTH_RATIO_OBJECT(self), FTH_RATIO_OBJECT(obj));
	return (BOOL_TO_FTH(flag == 0));
}

static void
rt_free(FTH self)
{
	mpr_free(FTH_RATIO_OBJECT(self));
}

static ficlRatio
mpr_new(void)
{
	ficlRatio 	rt;

	rt = mp_malloc(sizeof(mpr));
	mpr_init(rt);
	return (rt);
}

static void
mpr_free(ficlRatio rt)
{
	mpr_clear(rt);
	mp_free(rt);
}

static FTH
make_rational(ficlBignum num, ficlBignum den)
{
	ficlRatio	rt;

	rt = mpr_new();
	mpi_set(mpr_num(rt), num);
	mpi_set(mpr_den(rt), den);
	mpr_canonicalize(rt);
	return (fth_make_rational(rt));
}

FTH
fth_make_rational(ficlRatio rt)
{
	FTH 		self;

	self = fth_make_instance(ratio_tag, NULL);
	FTH_RATIO_OBJECT_SET(self, rt);
	return (self);
}

/*
 * Return a FTH ration object from NUM and DEN.
 */
FTH
fth_make_ratio(FTH num, FTH den)
{
#define h_make_ratio "( num den -- ratio )  return rational number\n\
123 456 make-ratio => 41/152\n\
355 113 make-ratio => 355/113\n\
Return a new ratio object with numerator NUM and denumerator DEN."
	if (den == FTH_ZERO) {
		FTH_MATH_ERROR_THROW("denominator 0");
		/* NOTREACHED */
		return (FTH_FALSE);
	}
	return (make_rational(fth_bignum_ref(num), fth_bignum_ref(den)));
}

FTH
fth_make_ratio_from_int(ficlInteger num, ficlInteger den)
{
	ficlRatio	rt;

	if (den == 0) {
		FTH_MATH_ERROR_THROW("denominator 0");
		/* NOTREACHED */
		return (FTH_FALSE);
	}
	rt = mpr_new();
	mpr_seti(rt, num, den);
	return (fth_make_rational(rt));
}

FTH
fth_make_ratio_from_float(ficlFloat f)
{
	ficlRatio	rt;

	rt = mpr_new();
	mpr_setd(rt, f);
	return (fth_make_rational(rt));
}

static void
ficl_q_dot(ficlVm *vm)
{
#define h_q_dot "( numb -- )  number output\n\
1.5 r. => 3/2\n\
Print rational number NUMB."
	FTH 		obj;

	FTH_STACK_CHECK(vm, 1, 0);
	obj = fth_pop_ficl_cell(vm);

	if (FTH_RATIO_P(obj))
		fth_printf("%S ", obj);
	else if (FTH_BIGNUM_P(obj))
		fth_printf("%S/1 ", obj);
	else {
		ficlFloat 	f;

		f = fth_float_ref(obj);
		fth_printf("%S ", fth_make_ratio_from_float(f));
	}
}

static void
ficl_qnegate(ficlVm *vm)
{
	ficlRatio 	r1;
	ficlRatio 	r2;

	FTH_STACK_CHECK(vm, 1, 1);
	r1 = ficlStackPopRatio(vm->dataStack);
	r2 = mpr_new();
	mpr_neg(r2, r1);
	mpr_free(r1);
	ficlStackPushRatio(vm->dataStack, r2);
}

/*
 * XXX: Don't remove this function, required by fth.m4 to set
 *	FTH_HAVE_RATIO=yes.
 */
FTH
fth_ratio_floor(FTH rt)
{
	ficlInteger 	i;

	if (FTH_RATIO_P(rt))
		i = (ficlInteger) FTH_FLOOR(mpr_getd(FTH_RATIO_OBJECT(rt)));
	else
		i = fth_int_ref(rt);

	return (fth_make_ratio_from_int(i, 1L));
}

static void
ficl_qfloor(ficlVm *vm)
{
#define h_qfloor "( x -- y )  y = floor(x) (ratio, result is int)"
	ficlRatio	x;
	ficlInteger 	y;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPopRatio(vm->dataStack);
	y = (ficlInteger) FTH_FLOOR(mpr_getd(x));
	mpr_free(x);
	ficlStackPushInteger(vm->dataStack, y);
}

static void
ficl_qceil(ficlVm *vm)
{
	ficlRatio	x;
	ficlInteger 	y;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPopRatio(vm->dataStack);
	y = (ficlInteger) FTH_CEIL(mpr_getd(x));
	mpr_free(x);
	ficlStackPushInteger(vm->dataStack, y);
}

static void
ficl_qabs(ficlVm *vm)
{
	ficlRatio 	x;
	ficlRatio 	y;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPopRatio(vm->dataStack);
	y = mpr_new();
	mpr_abs(y, x);
	mpr_free(x);
	ficlStackPushRatio(vm->dataStack, y);
}

static void
ficl_qinvert(ficlVm *vm)
{
#define h_qinvert "( x -- y )  y = 1/x (ratio)"
	ficlRatio 	x;
	ficlRatio 	y;

	FTH_STACK_CHECK(vm, 1, 1);
	x = ficlStackPopRatio(vm->dataStack);
	y = mpr_new();
	mpr_inv(y, x);
	mpr_free(x);
	ficlStackPushRatio(vm->dataStack, y);
}

static ficlRatio
rt_math(FTH m, FTH n, int type)
{
	ficlRatio 	x;
	ficlRatio 	y;
	ficlRatio 	z;

	x = fth_ratio_ref(m);
	y = fth_ratio_ref(n);
	z = mpr_new();

	switch (type) {
	case BN_ADD:
		mpr_add(z, x, y);
		break;
	case BN_SUB:
		mpr_sub(z, x, y);
		break;
	case BN_MUL:
		mpr_mul(z, x, y);
		break;
	case BN_DIV:
	default:
		mpr_div(z, x, y);
		break;
	}

	mpr_free(x);
	mpr_free(y);
	return (z);
}

static FTH
rt_add(FTH m, FTH n)
{
	return (fth_make_rational(rt_math(m, n, BN_ADD)));
}

static FTH
rt_sub(FTH m, FTH n)
{
	return (fth_make_rational(rt_math(m, n, BN_SUB)));
}

static FTH
rt_mul(FTH m, FTH n)
{
	return (fth_make_rational(rt_math(m, n, BN_MUL)));
}

static FTH
rt_div(FTH m, FTH n)
{
	return (fth_make_rational(rt_math(m, n, BN_DIV)));
}

#define N_RATIO_FUNC_TEST_ZERO(Name, OP)				\
static int								\
fth_rt_ ## Name(FTH m)							\
{									\
	int		flag;						\
									\
	if (FTH_RATIO_P(m))						\
		flag = (mpr_cmpi(FTH_RATIO_OBJECT(m), 0) OP 0);		\
	else if (FTH_BIGNUM_P(m))					\
		flag = (mpi_cmpi(FTH_BIGNUM_OBJECT(m), 0) OP 0);	\
	else {								\
		ficlRatio	x;					\
									\
		x = fth_ratio_ref(m);					\
		flag = (mpr_cmpi(x, 0) OP 0);				\
		mpr_free(x);						\
	}								\
	return (flag);							\
}									\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	int		flag;						\
									\
	FTH_STACK_CHECK(vm, 1, 1);					\
	flag = fth_rt_ ## Name(fth_pop_ficl_cell(vm));			\
	ficlStackPushBoolean(vm->dataStack, flag);			\
}									\
static char* h_ ## Name = "( x -- f )  x " #OP " 0 => flag (ratio)"

/*-
 * build: 
 *   int  fth_rt_qeqz(FTH m) ...    for C fth_number_equal_p etc
 *   void ficl_qeqz(ficlVm *vm) ... for Forth words
 */
N_RATIO_FUNC_TEST_ZERO(qeqz, ==);
N_RATIO_FUNC_TEST_ZERO(qnoteqz, !=);
N_RATIO_FUNC_TEST_ZERO(qlessz, <);
N_RATIO_FUNC_TEST_ZERO(qgreaterz, >);
N_RATIO_FUNC_TEST_ZERO(qlesseqz, <=);
N_RATIO_FUNC_TEST_ZERO(qgreatereqz, >=);

#define N_RATIO_FUNC_TEST_TWO_OP(Name, OP)				\
static int								\
fth_rt_ ## Name(FTH m, FTH n)						\
{									\
	int		flag;						\
									\
	if (FTH_RATIO_P(m)) {						\
		if (FTH_RATIO_P(n))					\
			flag = (mpr_cmp(FTH_RATIO_OBJECT(m),		\
			    FTH_RATIO_OBJECT(n)) OP 0);			\
		else {							\
			ficlRatio	y;				\
									\
			y = fth_ratio_ref(n);				\
			flag = (mpr_cmp(FTH_RATIO_OBJECT(m), y) OP 0);	\
			mpr_free(y);					\
		}							\
	} else if (FTH_RATIO_P(n)) {					\
		ficlRatio	x;					\
									\
		x = fth_ratio_ref(m);					\
		flag = (mpr_cmp(x, FTH_RATIO_OBJECT(n)) OP 0);		\
		mpr_free(x);						\
	} else {							\
		ficlRatio	x;					\
		ficlRatio	y;					\
									\
		x = fth_ratio_ref(m);					\
		y = fth_ratio_ref(n);					\
		flag = (mpr_cmp(x, y) OP 0);				\
		mpr_free(x);						\
		mpr_free(y);						\
	}								\
	return (flag);							\
}									\
static void								\
ficl_ ## Name(ficlVm *vm)						\
{									\
	FTH		m;						\
	FTH		n;						\
									\
	FTH_STACK_CHECK(vm, 2, 1);					\
	n = fth_pop_ficl_cell(vm);					\
	m = fth_pop_ficl_cell(vm);					\
	ficlStackPushBoolean(vm->dataStack, fth_rt_ ## Name(m, n));	\
}									\
static char* h_ ## Name = "( x y -- f )  x " #OP " y => flag (ratio)"

/*-
 * build: 
 *   int  fth_rt_qeq(FTH m, FTH n) ... for C fth_number_equal_p etc
 *   void ficl_qeq(ficlVm *vm) ...     for Forth words
 */
N_RATIO_FUNC_TEST_TWO_OP(qeq, ==);
N_RATIO_FUNC_TEST_TWO_OP(qnoteq, !=);
N_RATIO_FUNC_TEST_TWO_OP(qless, <);
N_RATIO_FUNC_TEST_TWO_OP(qgreater, >);
N_RATIO_FUNC_TEST_TWO_OP(qlesseq, <=);
N_RATIO_FUNC_TEST_TWO_OP(qgreatereq, >=);

N_BIGNUM_MATH_FUNC_OP(qadd, +, rt_add);
N_BIGNUM_MATH_FUNC_OP(qsub, -, rt_sub);
N_BIGNUM_MATH_FUNC_OP(qmul, *, rt_mul);
N_BIGNUM_MATH_FUNC_OP(qdiv, /, rt_div);

/*
 * Parse ficlRatio (in base 10) via xedit/lisp/mp (1/2, -3/2 etc).
 */
int
ficl_parse_ratio(ficlVm *vm, ficlString s)
{
	ficlRatio	rt;

	if (s.length < 3)
		return (FICL_FALSE);

	if (memchr(s.text, '/', s.length) == NULL)
		return (FICL_FALSE);
	
	rt = mpr_new();
	mpr_setstr(rt, s.text, 10);
	ficlStackPushRatio(vm->dataStack, rt);

	if (vm->state == FICL_VM_STATE_COMPILE)
		ficlPrimitiveLiteralIm(vm);

	return (FICL_TRUE);
}

static FTH
number_floor(FTH x)
{
	int 		type;
	ficlFloat 	f;

	if (x == 0 || !FTH_NUMBER_T_P(x)) {
		FTH_WRONG_NUMBER_TYPE(x, "a number");
		/* NOTREACHED */
		return (FTH_FALSE);
	}

	type = FTH_INSTANCE_TYPE(x);

	switch (type) {
	case FTH_FLOAT_T:
		return (fth_make_float(FTH_FLOOR(FTH_FLOAT_OBJECT(x))));
		break;
	case FTH_RATIO_T:
		f = FTH_FLOOR(mpr_getd(FTH_RATIO_OBJECT(x)));
		return (fth_make_ratio_from_float(f));
		break;
	default:
		FTH_WRONG_NUMBER_TYPE(x, "a ficlFloat or ficlRatio");
		break;
	}

	/* NOTREACHED */
	return (FTH_FALSE);
}

static FTH
number_inv(FTH x)
{
	int 		type;
	ficlRatio	res;

	if (x == 0 || !FTH_NUMBER_T_P(x)) {
		FTH_WRONG_NUMBER_TYPE(x, "a number");
		/* NOTREACHED */
		return (FTH_FALSE);
	}

	type = FTH_INSTANCE_TYPE(x);

	switch (type) {
	case FTH_RATIO_T:
		res = mpr_new();
		mpr_inv(res, FTH_RATIO_OBJECT(x));
		return (fth_make_rational(res));
		break;
	case FTH_FLOAT_T:
		return (fth_make_float(1.0 / FTH_FLOAT_OBJECT(x)));
		break;
	default:
		FTH_WRONG_NUMBER_TYPE(x, "a ficlFloat or ficlRatio");
		break;
	}

	/* NOTREACHED */
	return (FTH_FALSE);
}

/*
 * Return inexact number within ERR of X.
 */
FTH
fth_rationalize(FTH x, FTH err)
{
	if (FTH_INTEGER_P(x))
		return (x);

	if (FTH_RATIO_P(x) || FTH_INEXACT_P(x)) {
		ficlInteger 	a;
		ficlInteger 	a1;
		ficlInteger 	a2;
		ficlInteger 	b;
		ficlInteger 	b1;
		ficlInteger 	b2;
		ficlFloat 	fex;
		ficlFloat 	er;
		FTH 		ex;
		FTH 		dx;
		FTH 		rx;
		FTH 		tt;
		int 		i;

		if (FTH_RATIO_P(x))
			ex = x;
		else
			ex = fth_make_ratio_from_float(fth_float_ref(x));

		dx = number_floor(ex);

		if (fth_number_equal_p(dx, ex))
			return (ex);

		a1 = 0;
		a2 = 1;
		b1 = 1;
		b2 = 0;
		er = fabs(fth_float_ref(err));
		tt = FTH_ONE;
		i = 1000000;
		ex = fth_number_sub(ex, dx);

		if (ex == 0)
			return (FTH_ZERO);

		rx = number_inv(ex);
		fex = FTH_RATIO_REF_FLOAT(ex);

		while (--i) {
			a = a1 * fth_int_ref(tt) + a2;
			b = b1 * fth_int_ref(tt) + b2;

			if (b != 0 &&
			    fabs(fex - (ficlFloat) a / (ficlFloat) b) <= er)
				return (fth_number_add(dx,
					fth_make_ratio_from_int(a, b)));

			rx = number_inv(fth_number_sub(rx, tt));
			tt = number_floor(rx);
			a2 = a1;
			b2 = b1;
			a1 = a;
			b1 = b;
		}
	}
	return (FTH_ZERO);
}

static void
ficl_rationalize(ficlVm *vm)
{
#define h_rationalize "( x err -- val )  return number within ERR of X\n\
5.2  0.1  rationalize => 5.25\n\
5.4  0.1  rationalize => 5.5\n\
5.23 0.02 rationalize => 5.25\n\
Return inexact number within ERR of X."
	FTH 		x;
	FTH 		err;

	FTH_STACK_CHECK(vm, 2, 1);
	err = fth_pop_ficl_cell(vm);
	x = fth_pop_ficl_cell(vm);

	if (FTH_EXACT_P(x) && FTH_EXACT_P(err))
		fth_push_ficl_cell(vm, fth_rationalize(x, err));
	else {
		FTH 		rt;

		rt = fth_rationalize(x, err);
		ficlStackPushFTH(vm->dataStack, fth_exact_to_inexact(rt));
	}
}

#if HAVE_COMPLEX
#define N_CMP_COMPLEX_OP(Numb1, Numb2, Flag, OP) do {			\
	ficlComplex	x;						\
	ficlComplex	y;						\
									\
	x = fth_complex_ref(Numb1);					\
	y = fth_complex_ref(Numb2);					\
	Flag = (creal(x) OP creal(y));					\
									\
	if (Flag)							\
		Flag = (cimag(x) OP cimag(y));				\
} while (0)
#else				/* !HAVE_COMPLEX */
#define N_CMP_COMPLEX_OP(Numb1, Numb2, Flag, OP)
#endif				/* HAVE_COMPLEX */

#define N_CMP_BIGNUM_OP(Numb1, Numb2, Flag, OP, Name) do {		\
	Flag = fth_bn_b ## Name(Numb1, Numb2);				\
} while (0)

#define N_CMP_RATIO_OP(Numb1, Numb2, Flag, OP, Name) do {		\
	Flag = fth_rt_q ## Name(Numb1, Numb2);				\
} while (0)

#define N_CMP_TWO_OP(Numb1, Numb2, Flag, OP, Name) do {			\
	int		type;						\
									\
	type = -1;							\
									\
	if (FTH_NUMBER_T_P(Numb1))					\
		type = FTH_INSTANCE_TYPE(Numb1);			\
									\
	if (FTH_NUMBER_T_P(Numb2))					\
		type = FICL_MAX(type, (int)FTH_INSTANCE_TYPE(Numb2));	\
									\
	switch (type) {							\
	case FTH_FLOAT_T:						\
		Flag = (fth_float_ref(Numb1) OP fth_float_ref(Numb2));	\
		break;							\
	case FTH_COMPLEX_T:						\
		N_CMP_COMPLEX_OP(Numb1, Numb2, Flag, OP);		\
		break;							\
	case FTH_BIGNUM_T:						\
		N_CMP_BIGNUM_OP(Numb1, Numb2, Flag, OP, Name);		\
		break;							\
	case FTH_RATIO_T:						\
		N_CMP_RATIO_OP(Numb1, Numb2, Flag, OP, Name);		\
		break;							\
	case FTH_LLONG_T:						\
		Flag = (fth_long_long_ref(Numb1) OP 			\
		    fth_long_long_ref(Numb2));				\
		break;							\
	default:							\
		Flag = (Numb1 OP Numb2);				\
		break;							\
	}								\
} while (0)

int
fth_number_equal_p(FTH m, FTH n)
{
	int 		flag;

	if (NUMB_FIXNUM_P(m) && NUMB_FIXNUM_P(n))
		return (FIX_TO_INT(m) == FIX_TO_INT(n));

	N_CMP_TWO_OP(m, n, flag, ==, eq);
	return (flag);
}

int
fth_number_less_p(FTH m, FTH n)
{
	int 		flag;

	if (NUMB_FIXNUM_P(m) && NUMB_FIXNUM_P(n))
		return (FIX_TO_INT(m) < FIX_TO_INT(n));

	N_CMP_TWO_OP(m, n, flag, <, less);
	return (flag);
}

#if HAVE_COMPLEX
#define N_MATH_COMPLEX_OP(N1, N2, OP)					\
	N1 = fth_make_complex(fth_complex_ref(N1) OP fth_complex_ref(N2))
#else				/* !HAVE_COMPLEX */
#define N_MATH_COMPLEX_OP(N1, N2, OP)
#endif				/* HAVE_COMPLEX */

#define N_MATH_BIGNUM_OP(Numb1, Numb2, GOP) do {			\
	Numb1 = bn_ ## GOP(Numb1, Numb2);				\
} while (0)

#define N_MATH_RATIO_OP(Numb1, Numb2, GOP) do {				\
	Numb1 = rt_ ## GOP(Numb1, Numb2);				\
} while (0)

#define N_MATH_OP(Numb1, Numb2, OP, GOP) do {				\
	int		type;						\
									\
	type = -1;							\
									\
	if (FTH_NUMBER_T_P(Numb1))					\
		type = FTH_INSTANCE_TYPE(Numb1);			\
									\
	if (FTH_NUMBER_T_P(Numb2))					\
		type = FICL_MAX(type, (int)FTH_INSTANCE_TYPE(Numb2));	\
									\
	switch (type) {							\
	case FTH_FLOAT_T:						\
		Numb1 = fth_make_float(fth_float_ref(Numb1) OP 		\
		    fth_float_ref(Numb2));				\
		break;							\
	case FTH_COMPLEX_T:						\
		N_MATH_COMPLEX_OP(Numb1, Numb2, OP);			\
		break;							\
	case FTH_BIGNUM_T:						\
		N_MATH_BIGNUM_OP(Numb1, Numb2, GOP);			\
		break;							\
	case FTH_RATIO_T:						\
		N_MATH_RATIO_OP(Numb1, Numb2, GOP);			\
		break;							\
	case FTH_LLONG_T:						\
		Numb1 = fth_make_long_long(fth_long_long_ref(Numb1) OP	\
		    fth_long_long_ref(Numb2));				\
		break;							\
	default:							\
		Numb1 = Numb1 OP Numb2;					\
		break;							\
	}								\
} while (0)

FTH
fth_number_add(FTH m, FTH n)
{
	if (NUMB_FIXNUM_P(m) && NUMB_FIXNUM_P(n))
		return (fth_make_int(FIX_TO_INT(m) + FIX_TO_INT(n)));

	N_MATH_OP(m, n, +, add);
	return (m);
}

FTH
fth_number_sub(FTH m, FTH n)
{
	if (NUMB_FIXNUM_P(m) && NUMB_FIXNUM_P(n))
		return (fth_make_int(FIX_TO_INT(m) - FIX_TO_INT(n)));

	/* suggested from scan-build */
	if (m == 0 || n == 0)
		return (m);

	N_MATH_OP(m, n, -, sub);
	return (m);
}

FTH
fth_number_mul(FTH m, FTH n)
{
	if (NUMB_FIXNUM_P(m) && NUMB_FIXNUM_P(n))
		return (fth_make_int(FIX_TO_INT(m) * FIX_TO_INT(n)));

	N_MATH_OP(m, n, *, mul);
	return (m);
}

FTH
fth_number_div(FTH m, FTH n)
{
	if (NUMB_FIXNUM_P(m) && NUMB_FIXNUM_P(n))
		return (fth_make_int(FIX_TO_INT(m) / FIX_TO_INT(n)));

	N_MATH_OP(m, n, /, div);
	return (m);
}

FTH
fth_exact_to_inexact(FTH obj)
{
#define h_exact_to_inexact "( numb1 -- numb2 )  convert to inexact number\n\
3/2 exact->inexact => 1.5\n\
Convert NUMB to an inexact number.\n\
See also inexact->exact."
	FTH_ASSERT_ARGS(FTH_NUMBER_P(obj), obj, FTH_ARG1, "a number");
	if (FTH_EXACT_P(obj))
		return (fth_make_float(fth_float_ref(obj)));
	return (obj);
}

FTH
fth_inexact_to_exact(FTH obj)
{
#define h_inexact_to_exact "( numb1 -- numb2 )  convert to exact number\n\
1.5 inexact->exact => 3/2\n\
Convert NUMB to an exact number.\n\
See also exact->inexact."
	FTH_ASSERT_ARGS(FTH_NUMBER_P(obj), obj, FTH_ARG1, "a number");
	if (FTH_INEXACT_P(obj))
		return (fth_make_ratio_from_float(fth_float_ref(obj)));
	return (obj);
}

/*
 * Return numerator from OBJ or 0.
 */
FTH
fth_numerator(FTH obj)
{
#define h_numerator "( obj -- numerator )  return numerator\n\
3/4 numerator => 3\n\
5 numerator => 5\n\
1.5 numerator => 0\n\
Return numerator of OBJ or 0.\n\
See also denominator."
	ficlBignum 	res;

	if (FTH_INTEGER_P(obj))
		return (obj);

	if (!FTH_RATIO_P(obj))
		return (FTH_ZERO);

	if (mpi_fiti(FTH_RATIO_NUM(obj))) {
		long	x;

		x = mpi_geti(FTH_RATIO_NUM(obj));
		return (fth_make_int(x));
	}
	res = mpi_new();
	mpi_set(res, FTH_RATIO_NUM(obj));
	return (fth_make_bignum(res));
}

/*
 * Return denominator from OBJ or 1.
 */
FTH
fth_denominator(FTH obj)
{
#define h_denominator "( obj -- denominator )  return denominator\n\
3/4 denominator => 4\n\
5 denominator => 1\n\
1.5 denominator => 1\n\
Return denominator of OBJ or 1.\n\
See also numerator."
	ficlBignum 	res;

	if (!FTH_RATIO_P(obj))
		return (FTH_ONE);

	if (mpi_fiti(FTH_RATIO_DEN(obj))) {
		long	x;

		x = mpi_geti(FTH_RATIO_DEN(obj));
		return (fth_make_int(x));
	}
	res = mpi_new();
	mpi_set(res, FTH_RATIO_DEN(obj));
	return (fth_make_bignum(res));
}

static void
ficl_odd_p(ficlVm *vm)
{
#define h_odd_p "( numb -- f )  test if NUMB is odd\n\
3 odd? => #t\n\
6 odd? => #f\n\
Return #t if NUMB is odd, otherwise #f.\n\
See also even?"
	FTH 		m;
	int 		flag;

	FTH_STACK_CHECK(vm, 1, 1);
	m = fth_pop_ficl_cell(vm);
	flag = ((fth_int_ref(m) % 2) != 0);
	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_even_p(ficlVm *vm)
{
#define h_even_p "( numb -- f )  test if NUMB is even\n\
3 even? => #f\n\
6 even? => #t\n\
Return #t if NUMB is even, otherwise #f.\n\
See also odd?"
	FTH 		m;
	int 		flag;

	FTH_STACK_CHECK(vm, 1, 1);
	m = fth_pop_ficl_cell(vm);
	flag = ((fth_int_ref(m) % 2) == 0);
	ficlStackPushBoolean(vm->dataStack, flag);
}

static void
ficl_prime_p(ficlVm *vm)
{
#define h_prime_p "( numb -- f )  test if NUMB is a prime number\n\
3   prime? => #t\n\
123 prime? => #f\n\
Return #t if NUMB is a prime number, otherwise #f."
	FTH 		m;
	int 		flag;
	ficl2Integer 	x;

	FTH_STACK_CHECK(vm, 1, 1);
	m = fth_pop_ficl_cell(vm);
	x = fth_long_long_ref(m);
	flag = 0;

	if (x == 2)
		flag = 1;
	else if ((x % 2) != 0) {
		int 		i;

		for (i = 3; i < (int) sqrt((double) x); i += 2)
			if (((x % i) == 0)) {
				flag = 0;
				goto finish;
			}
		flag = 1;
	}
finish:
	ficlStackPushBoolean(vm->dataStack, flag);
}

/*-
 * fesetround(3) may use one of the following constants:
 *
 * FE_TONEAREST
 * FE_DOWNWARD
 * FE_UPWARD
 * FE_TOWARDZERO
 */
#if defined(HAVE_FENV_H)
#include <fenv.h>
#endif

static void
ficl_fegetround(ficlVm *vm)
{
#define h_fegetround "( -- n )  float rounding mode\
Return current floating-point rounding mode,  one of:\n\
FE_TONEAREST\n\
FE_DOWNWARD\n\
FE_UPWARD\n\
FE_TOWARDZERO\n\
See also fesetround."
	ficlInteger	n;

	FTH_STACK_CHECK(vm, 0, 1);
#if defined(HAVE_FEGETROUND)
	n = fegetround();
#else
	n = -1;
#endif
	ficlStackPushInteger(vm->dataStack, n);
}

static void
ficl_fesetround(ficlVm *vm)
{
#define h_fesetround "( n -- )  set float rounding mode\
Set current floating-point rounding mode,  one of:\n\
FE_TONEAREST\n\
FE_DOWNWARD\n\
FE_UPWARD\n\
FE_TOWARDZERO\n\
See also fegetround."
	ficlInteger	n;

	FTH_STACK_CHECK(vm, 1, 0);
	n = ficlStackPopInteger(vm->dataStack);

#if defined(HAVE_FESETROUND)
	if (fesetround(n) < 0)
		fth_warning("%d not supported, nothing changed", n);
#endif
}

void
init_number_types(void)
{
	/* init llong */
	llong_tag = make_object_number_type(FTH_STR_LLONG,
	    FTH_LLONG_T, N_EXACT_T);
	fth_set_object_inspect(llong_tag, ll_inspect);
	fth_set_object_to_string(llong_tag, ll_to_string);
	fth_set_object_copy(llong_tag, ll_copy);
	fth_set_object_equal_p(llong_tag, ll_equal_p);

	/* init float */
	float_tag = make_object_number_type(FTH_STR_FLOAT,
	    FTH_FLOAT_T, N_INEXACT_T);
	fth_set_object_inspect(float_tag, fl_inspect);
	fth_set_object_to_string(float_tag, fl_to_string);
	fth_set_object_copy(float_tag, fl_copy);
	fth_set_object_equal_p(float_tag, fl_equal_p);

#if HAVE_COMPLEX
	/* complex */
	complex_tag = make_object_number_type(FTH_STR_COMPLEX,
	    FTH_COMPLEX_T, N_INEXACT_T);
	fth_set_object_inspect(complex_tag, cp_inspect);
	fth_set_object_to_string(complex_tag, cp_to_string);
	fth_set_object_copy(complex_tag, cp_copy);
	fth_set_object_equal_p(complex_tag, cp_equal_p);
#endif				/* HAVE_COMPLEX */

	/* init bignum */
	bignum_tag = make_object_number_type(FTH_STR_BIGNUM,
	    FTH_BIGNUM_T, N_EXACT_T);
	fth_set_object_inspect(bignum_tag, bn_inspect);
	fth_set_object_to_string(bignum_tag, bn_to_string);
	fth_set_object_copy(bignum_tag, bn_copy);
	fth_set_object_equal_p(bignum_tag, bn_equal_p);
	fth_set_object_free(bignum_tag, bn_free);

	/* init ratio */
	ratio_tag = make_object_number_type(FTH_STR_RATIO,
	    FTH_RATIO_T, N_EXACT_T);
	fth_set_object_inspect(ratio_tag, rt_inspect);
	fth_set_object_to_string(ratio_tag, rt_to_string);
	fth_set_object_copy(ratio_tag, rt_copy);
	fth_set_object_equal_p(ratio_tag, rt_equal_p);
	fth_set_object_free(ratio_tag, rt_free);
}

#if defined(HAVE_SYS_TIME_H)
#include <sys/time.h>
#endif
#if defined(HAVE_TIME_H)
#include <time.h>
#endif

void
init_number(void)
{
	ficlDictionary *env;
#if !defined(INFINITY)
	double 		tmp, inf;

	inf = tmp = 1e+10;
	while (1) {
		inf *= 1e+10;
		if (inf == tmp)
			break;
		tmp = inf;
	}
	fth_infinity = inf;
#endif

	/* int, llong, rand */
	fth_srand((ficlUnsigned) time(NULL));
	FTH_PRI1("number?", ficl_number_p, h_number_p);
	FTH_PRI1("fixnum?", ficl_fixnum_p, h_fixnum_p);
	FTH_PRI1("unsigned?", ficl_unsigned_p, h_unsigned_p);
	FTH_PRI1("long-long?", ficl_llong_p, h_llong_p);
	FTH_PRI1("off-t?", ficl_llong_p, h_llong_p);
	FTH_PRI1("ulong-long?", ficl_ullong_p, h_ullong_p);
	FTH_PRI1("uoff-t?", ficl_ullong_p, h_ullong_p);
	FTH_PRI1("integer?", ficl_integer_p, h_integer_p);
	FTH_PRI1("exact?", ficl_exact_p, h_exact_p);
	FTH_PRI1("inexact?", ficl_inexact_p, h_inexact_p);
	FTH_PRI1("make-long-long", ficl_to_d, h_to_d);
	FTH_PRI1(">llong", ficl_to_d, h_to_d);
	FTH_PRI1("make-off-t", ficl_to_d, h_to_d);
	FTH_PRI1("make-ulong-long", ficl_to_ud, h_to_ud);
	FTH_PRI1("rand-seed-ref", ficl_rand_seed_ref, h_rand_seed_ref);
	FTH_PRI1("rand-seed-set!", ficl_rand_seed_set, h_rand_seed_set);
	FTH_PRI1("random", ficl_random, h_random);
	FTH_PRI1("frandom", ficl_frandom, h_frandom);
	FTH_PRI1(".r", ficl_dot_r, h_dot_r);
	FTH_PRI1("u.r", ficl_u_dot_r, h_u_dot_r);
	FTH_PRI1("d.", ficl_d_dot, h_d_dot);
	FTH_PRI1("ud.", ficl_ud_dot, h_ud_dot);
	FTH_PRI1("d.r", ficl_d_dot_r, h_d_dot_r);
	FTH_PRI1("ud.r", ficl_ud_dot_r, h_ud_dot_r);
	FTH_PRI1("u=", ficl_ueq, h_ueq);
	FTH_PRI1("u<>", ficl_unoteq, h_unoteq);
	FTH_PRI1("u<", ficl_uless, h_uless);
	FTH_PRI1("u<=", ficl_ulesseq, h_ulesseq);
	FTH_PRI1("u>", ficl_ugreater, h_ugreater);
	FTH_PRI1("u>=", ficl_ugreatereq, h_ugreatereq);
	FTH_PRI1("s>d", ficl_to_d, h_to_d);
	FTH_PRI1("s>ud", ficl_to_ud, h_to_ud);
	FTH_PRI1("d>s", ficl_to_s, h_to_s);
	FTH_PRI1("f>d", ficl_to_d, h_to_d);
	FTH_PRI1("f>ud", ficl_to_ud, h_to_ud);
	FTH_PRI1("d>f", ficl_to_f, h_to_f);
	FTH_PRI1("dzero?", ficl_dzero, h_dzero);
	FTH_PRI1("d0=", ficl_dzero, h_dzero);
	FTH_PRI1("d0<>", ficl_dnotz, h_dnotz);
	FTH_PRI1("d0<", ficl_dlessz, h_dlessz);
	FTH_PRI1("dnegative?", ficl_dlessz, h_dlessz);
	FTH_PRI1("d0<=", ficl_dlesseqz, h_dlesseqz);
	FTH_PRI1("d0>", ficl_dgreaterz, h_dgreaterz);
	FTH_PRI1("d0>=", ficl_dgreatereqz, h_dgreatereqz);
	FTH_PRI1("dpositive?", ficl_dgreatereqz, h_dgreatereqz);
	FTH_PRI1("d=", ficl_deq, h_deq);
	FTH_PRI1("d<>", ficl_dnoteq, h_dnoteq);
	FTH_PRI1("d<", ficl_dless, h_dless);
	FTH_PRI1("d<=", ficl_dlesseq, h_dlesseq);
	FTH_PRI1("d>", ficl_dgreater, h_dgreater);
	FTH_PRI1("d>=", ficl_dgreatereq, h_dgreatereq);
	FTH_PRI1("du=", ficl_dueq, h_dueq);
	FTH_PRI1("du<>", ficl_dunoteq, h_dunoteq);
	FTH_PRI1("du<", ficl_duless, h_duless);
	FTH_PRI1("du<=", ficl_dulesseq, h_dulesseq);
	FTH_PRI1("du>", ficl_dugreater, h_dugreater);
	FTH_PRI1("du>=", ficl_dugreatereq, h_dugreatereq);
	FTH_PRI1("d+", ficl_dadd, h_dadd);
	FTH_PRI1("d-", ficl_dsub, h_dsub);
	FTH_PRI1("d*", ficl_dmul, h_dmul);
	FTH_PRI1("d/", ficl_ddiv, h_ddiv);
	FTH_PRI1("dnegate", ficl_dnegate, h_dnegate);
	FTH_PRI1("dabs", ficl_dabs, h_dabs);
	FTH_PRI1("dmin", ficl_dmin, h_dmin);
	FTH_PRI1("dmax", ficl_dmax, h_dmax);
	FTH_PRI1("d2*", ficl_dtwostar, h_dtwostar);
	FTH_PRI1("d2/", ficl_dtwoslash, h_dtwoslash);
	FTH_ADD_FEATURE_AND_INFO(FTH_STR_LLONG, h_list_of_llong_functions);

	/* float */
	FTH_PRI1("float?", ficl_float_p, h_float_p);
	FTH_PRI1("inf?", ficl_inf_p, h_inf_p);
	FTH_PRI1("nan?", ficl_nan_p, h_nan_p);
	FTH_PRI1("inf", ficl_inf, h_inf);
	FTH_PRI1("nan", ficl_nan, h_nan);
	FTH_PRI1("f.r", ficl_f_dot_r, h_f_dot_r);
	FTH_PRI1("uf.r", ficl_uf_dot_r, h_uf_dot_r);
	FTH_PRI1("floats", ficl_dfloats, h_dfloats);
	FTH_PRI1("sfloats", ficl_dfloats, h_dfloats);
	FTH_PRI1("dfloats", ficl_dfloats, h_dfloats);
	FTH_PRI1("falign", ficl_falign, h_falign);
	FTH_PRI1("f>s", ficl_to_s, h_to_s);
	FTH_PRI1("s>f", ficl_to_f, h_to_f);
	FTH_PRI1("f**", ficl_fpow, h_fpow);
	FTH_PRI1("fpow", ficl_fpow, h_fpow);
	FTH_PRI1("fabs", ficl_fabs, h_fabs);
#if !HAVE_COMPLEX
	FTH_PRI1("magnitude", ficl_fabs, h_fabs);
#endif
	FTH_PRI1("fmod", ficl_fmod, h_fmod);
	FTH_PRI1("floor", ficl_floor, h_floor);
	FTH_PRI1("fceil", ficl_fceil, h_fceil);
	FTH_PRI1("ftrunc", ficl_ftrunc, h_ftrunc);
	FTH_PRI1("fround", ficl_fround, h_fround);
	FTH_PRI1("fsqrt", ficl_fsqrt, h_fsqrt);
	FTH_PRI1("fexp", ficl_fexp, h_fexp);
	FTH_PRI1("fexpm1", ficl_fexpm1, h_fexpm1);
	FTH_PRI1("flog", ficl_flog, h_flog);
	FTH_PRI1("flogp1", ficl_flogp1, h_flogp1);
	FTH_PRI1("flog1p", ficl_flogp1, h_flogp1);
	FTH_PRI1("flog2", ficl_flog2, h_flog2);
	FTH_PRI1("flog10", ficl_flog10, h_flog10);
	FTH_PRI1("falog", ficl_falog, h_falog);
	FTH_PRI1("fsin", ficl_fsin, h_fsin);
	FTH_PRI1("fcos", ficl_fcos, h_fcos);
	FTH_PRI1("fsincos", ficl_fsincos, h_fsincos);
	FTH_PRI1("ftan", ficl_ftan, h_ftan);
	FTH_PRI1("fasin", ficl_fasin, h_fasin);
	FTH_PRI1("facos", ficl_facos, h_facos);
	FTH_PRI1("fatan", ficl_fatan, h_fatan);
	FTH_PRI1("fatan2", ficl_fatan2, h_fatan2);
	FTH_PRI1("fsinh", ficl_fsinh, h_fsinh);
	FTH_PRI1("fcosh", ficl_fcosh, h_fcosh);
	FTH_PRI1("ftanh", ficl_ftanh, h_ftanh);
	FTH_PRI1("fasinh", ficl_fasinh, h_fasinh);
	FTH_PRI1("facosh", ficl_facosh, h_facosh);
	FTH_PRI1("fatanh", ficl_fatanh, h_fatanh);

	/* math.h */
#if !defined(M_E)
#define M_E			2.7182818284590452354	/* e */
#endif
#if !defined(M_LN2)
#define M_LN2			0.69314718055994530942	/* log(2) */
#endif
#if !defined(M_LN10)
#define M_LN10			2.30258509299404568402	/* log(10) */
#endif
#if !defined(M_PI)
#define M_PI			3.14159265358979323846	/* pi */
#endif
#if !defined(M_PI_2)
#define M_PI_2			1.57079632679489661923	/* pi/2 */
#endif
#if !defined(M_TWO_PI)
#define M_TWO_PI		(M_PI * 2.0)	/* pi*2 */
#endif
#if !defined(M_SQRT2)
#define M_SQRT2			1.41421356237309504880	/* sqrt(2) */
#endif
	fth_define("euler", fth_make_float(M_E));
	fth_define("ln-two", fth_make_float(M_LN2));
	fth_define("ln-ten", fth_make_float(M_LN10));
	fth_define("pi", fth_make_float(M_PI));
	fth_define("two-pi", fth_make_float(M_TWO_PI));
	fth_define("half-pi", fth_make_float(M_PI_2));
	fth_define("sqrt-two", fth_make_float(M_SQRT2));
	FTH_ADD_FEATURE_AND_INFO(FTH_STR_FLOAT, h_list_of_float_functions);

	/* complex */
	FTH_PRI1("complex?", ficl_complex_p, h_complex_p);
	FTH_PRI1("real-ref", ficl_creal, h_creal);
	FTH_PRI1("imag-ref", ficl_cimage, h_cimage);
	FTH_PRI1("image-ref", ficl_cimage, h_cimage);
#if HAVE_COMPLEX
	FTH_PRI1("Complex-I", ficl_complex_i, h_complex_i);
	FTH_PRI1("cnegate", ficl_cnegate, h_cnegate);
	FTH_PRI1("make-rectangular", ficl_make_complex_rectangular,
	    h_make_complex_rectangular);
	FTH_PRI1(">complex", ficl_make_complex_rectangular,
	    h_make_complex_rectangular);
	FTH_PRI1("make-polar", ficl_make_complex_polar,
	    h_make_complex_polar);
	FTH_PRI1("c.", ficl_c_dot, h_c_dot);
	FTH_PRI1("s>c", ficl_to_c, h_to_c);
	FTH_PRI1("c>s", ficl_to_s, h_to_s);
	FTH_PRI1("f>c", ficl_to_c, h_to_c);
	FTH_PRI1("c>f", ficl_to_f, h_to_f);
	FTH_PRI1("q>c", ficl_to_c, h_to_c);
	FTH_PRI1("r>c", ficl_to_c, h_to_c);
	FTH_PRI1(">c", ficl_to_c, h_to_c);
	FTH_PRI1("c0=", ficl_ceqz, h_ceqz);
	FTH_PRI1("c0<>", ficl_cnoteqz, h_cnoteqz);
	FTH_PRI1("c=", ficl_ceq, h_ceq);
	FTH_PRI1("c<>", ficl_cnoteq, h_cnoteq);
	FTH_PRI1("c+", ficl_cadd, h_cadd);
	FTH_PRI1("c-", ficl_csub, h_csub);
	FTH_PRI1("c*", ficl_cmul, h_cmul);
	FTH_PRI1("c/", ficl_cdiv, h_cdiv);
	FTH_PRI1("1/c", ficl_creciprocal, h_creciprocal);
	FTH_PRI1("carg", ficl_carg, h_carg);
	FTH_PRI1("cabs", ficl_cabs, h_cabs);
	FTH_PRI1("magnitude", ficl_cabs, h_cabs);
	FTH_PRI1("cabs2", ficl_cabs2, h_cabs2);
	FTH_PRI1("c**", ficl_cpow, h_cpow);
	FTH_PRI1("cpow", ficl_cpow, h_cpow);
	FTH_PRI1("conj", ficl_cconj, h_cconj);
	FTH_PRI1("conjugate", ficl_cconj, h_cconj);
	FTH_PRI1("csqrt", ficl_csqrt, h_csqrt);
	FTH_PRI1("cexp", ficl_cexp, h_cexp);
	FTH_PRI1("clog", ficl_clog, h_clog);
	FTH_PRI1("clog10", ficl_clog10, h_clog10);
	FTH_PRI1("csin", ficl_csin, h_csin);
	FTH_PRI1("ccos", ficl_ccos, h_ccos);
	FTH_PRI1("ctan", ficl_ctan, h_ctan);
	FTH_PRI1("casin", ficl_casin, h_casin);
	FTH_PRI1("cacos", ficl_cacos, h_cacos);
	FTH_PRI1("catan", ficl_catan, h_catan);
	FTH_PRI1("catan2", ficl_catan2, h_catan2);
	FTH_PRI1("csinh", ficl_csinh, h_csinh);
	FTH_PRI1("ccosh", ficl_ccosh, h_ccosh);
	FTH_PRI1("ctanh", ficl_ctanh, h_ctanh);
	FTH_PRI1("casinh", ficl_casinh, h_casinh);
	FTH_PRI1("cacosh", ficl_cacosh, h_cacosh);
	FTH_PRI1("catanh", ficl_catanh, h_catanh);
	FTH_ADD_FEATURE_AND_INFO(FTH_STR_COMPLEX, h_list_of_complex_functions);
#endif				/* HAVE_COMPLEX */

	/* bignum */
	FTH_PRI1("bignum?", ficl_bignum_p, h_bignum_p);
	FTH_PRI1("make-bignum", ficl_to_bn, h_to_bn);
	FTH_PRI1(">bignum", ficl_to_bn, h_to_bn);
	FTH_PRI1("bn.", ficl_bn_dot, h_bn_dot);
	FTH_PRI1("s>b", ficl_to_bn, h_to_bn);
	FTH_PRI1("b>s", ficl_to_s, h_to_s);
	FTH_PRI1("f>b", ficl_to_bn, h_to_bn);
	FTH_PRI1("b>f", ficl_to_f, h_to_f);
	FTH_PRI1("b0=", ficl_beqz, h_beqz);
	FTH_PRI1("b0<>", ficl_bnoteqz, h_bnoteqz);
	FTH_PRI1("b0<", ficl_blessz, h_blessz);
	FTH_PRI1("b0>", ficl_bgreaterz, h_bgreaterz);
	FTH_PRI1("b0<=", ficl_blesseqz, h_blesseqz);
	FTH_PRI1("b0>=", ficl_bgreatereqz, h_bgreatereqz);
	FTH_PRI1("b=", ficl_beq, h_beq);
	FTH_PRI1("b<>", ficl_bnoteq, h_bnoteq);
	FTH_PRI1("b<", ficl_bless, h_bless);
	FTH_PRI1("b>", ficl_bgreater, h_bgreater);
	FTH_PRI1("b<=", ficl_blesseq, h_blesseq);
	FTH_PRI1("b>=", ficl_bgreatereq, h_bgreatereq);
	FTH_PRI1("b+", ficl_badd, h_badd);
	FTH_PRI1("b-", ficl_bsub, h_bsub);
	FTH_PRI1("b*", ficl_bmul, h_bmul);
	FTH_PRI1("b/", ficl_bdiv, h_bdiv);
	FTH_PRI1("bgcd", ficl_bgcd, h_bgcd);
	FTH_PRI1("blcm", ficl_blcm, h_blcm);
	FTH_PRI1("b**", ficl_bpow, h_bpow);
	FTH_PRI1("bpow", ficl_bpow, h_bpow);
	FTH_PRI1("broot", ficl_broot, h_broot);
	FTH_PRI1("bsqrt", ficl_bsqrt, h_bsqrt);
	FTH_PRI1("bnegate", ficl_bnegate, h_dnegate);
	FTH_PRI1("babs", ficl_babs, h_dabs);
	FTH_PRI1("bmin", ficl_bmin, h_dmin);
	FTH_PRI1("bmax", ficl_bmax, h_dmax);
	FTH_PRI1("b2*", ficl_btwostar, h_dtwostar);
	FTH_PRI1("b2/", ficl_btwoslash, h_dtwoslash);
	FTH_PRI1("bmod", ficl_bmod, h_bmod);
	FTH_PRI1("b/mod", ficl_bslashmod, h_bslashmod);
	FTH_PRI1("blshift", ficl_blshift, h_blshift);
	FTH_PRI1("brshift", ficl_brshift, h_brshift);
	FTH_ADD_FEATURE_AND_INFO(FTH_STR_BIGNUM, h_list_of_bignum_functions);

	/* ratio */
	FTH_PRI1("ratio?", ficl_ratio_p, h_ratio_p);
	FTH_PRI1("rational?", ficl_ratio_p, h_ratio_p);
	FTH_PROC("make-ratio", fth_make_ratio, 2, 0, 0, h_make_ratio);
	FTH_PRI1(">ratio", ficl_to_rt, h_to_rt);
	FTH_PRI1("rationalize", ficl_rationalize, h_rationalize);
	FTH_PRI1("q.", ficl_q_dot, h_q_dot);
	FTH_PRI1("r.", ficl_q_dot, h_q_dot);
	FTH_PRI1("s>q", ficl_to_rt, h_to_rt);
	FTH_PRI1("s>r", ficl_to_rt, h_to_rt);
	FTH_PRI1("q>s", ficl_to_s, h_to_s);
	FTH_PRI1("r>s", ficl_to_s, h_to_s);
	FTH_PRI1("c>q", ficl_to_rt, h_to_rt);
	FTH_PRI1("c>r", ficl_to_rt, h_to_rt);
	FTH_PRI1("f>q", ficl_to_rt, h_to_rt);
	FTH_PRI1("f>r", ficl_to_rt, h_to_rt);
	FTH_PRI1("q>f", ficl_to_f, h_to_f);
	FTH_PRI1("r>f", ficl_to_f, h_to_f);
	FTH_PRI1("q0=", ficl_qeqz, h_qeqz);
	FTH_PRI1("q0<>", ficl_qnoteqz, h_qnoteqz);
	FTH_PRI1("q0<", ficl_qlessz, h_qlessz);
	FTH_PRI1("q0>", ficl_qgreaterz, h_qgreaterz);
	FTH_PRI1("q0<=", ficl_qlesseqz, h_qlesseqz);
	FTH_PRI1("q0>=", ficl_qgreatereqz, h_qgreatereqz);
	FTH_PRI1("q=", ficl_qeq, h_qeq);
	FTH_PRI1("q<>", ficl_qnoteq, h_qnoteq);
	FTH_PRI1("q<", ficl_qless, h_qless);
	FTH_PRI1("q>", ficl_qgreater, h_qgreater);
	FTH_PRI1("q<=", ficl_qlesseq, h_qlesseq);
	FTH_PRI1("q>=", ficl_qgreatereq, h_qgreatereq);
	FTH_PRI1("q+", ficl_qadd, h_qadd);
	FTH_PRI1("q-", ficl_qsub, h_qsub);
	FTH_PRI1("q*", ficl_qmul, h_qmul);
	FTH_PRI1("q/", ficl_qdiv, h_qdiv);
	FTH_PRI1("r+", ficl_qadd, h_qadd);
	FTH_PRI1("r-", ficl_qsub, h_qsub);
	FTH_PRI1("r*", ficl_qmul, h_qmul);
	FTH_PRI1("r/", ficl_qdiv, h_qdiv);
	FTH_PRI1("q**", ficl_fpow, h_fpow);
	FTH_PRI1("qpow", ficl_fpow, h_fpow);
	FTH_PRI1("r**", ficl_fpow, h_fpow);
	FTH_PRI1("rpow", ficl_fpow, h_fpow);
	FTH_PRI1("qnegate", ficl_qnegate, h_dnegate);
	FTH_PRI1("rnegate", ficl_qnegate, h_dnegate);
	FTH_PRI1("qfloor", ficl_qfloor, h_qfloor);
	FTH_PRI1("rfloor", ficl_qfloor, h_qfloor);
	FTH_PRI1("qceil", ficl_qceil, h_fceil);
	FTH_PRI1("rceil", ficl_qceil, h_fceil);
	FTH_PRI1("qabs", ficl_qabs, h_dabs);
	FTH_PRI1("rabs", ficl_qabs, h_dabs);
	FTH_PRI1("1/q", ficl_qinvert, h_qinvert);
	FTH_PRI1("1/r", ficl_qinvert, h_qinvert);
	FTH_ADD_FEATURE_AND_INFO(FTH_STR_RATIO, h_list_of_ratio_functions);

	FTH_PROC("exact->inexact", fth_exact_to_inexact, 1, 0, 0,
	    h_exact_to_inexact);
	FTH_PROC("inexact->exact", fth_inexact_to_exact, 1, 0, 0,
	    h_inexact_to_exact);
	FTH_PROC("numerator", fth_numerator, 1, 0, 0, h_numerator);
	FTH_PROC("denominator", fth_denominator, 1, 0, 0, h_denominator);
	FTH_PRI1("odd?", ficl_odd_p, h_odd_p);
	FTH_PRI1("even?", ficl_even_p, h_even_p);
	FTH_PRI1("prime?", ficl_prime_p, h_prime_p);

	/* fenv(3), fegetround(3), fesetround(3) */
	FTH_PRI1("fegetround", ficl_fegetround, h_fegetround);
	FTH_PRI1("fesetround", ficl_fesetround, h_fesetround);
#if defined(HAVE_FENV_H)
	FTH_SET_CONSTANT(FE_TONEAREST);
	FTH_SET_CONSTANT(FE_DOWNWARD);
	FTH_SET_CONSTANT(FE_UPWARD);
	FTH_SET_CONSTANT(FE_TOWARDZERO);
#endif

	/* From ficlSystemCompileCore(), ficl/primitive.c */
	env = ficlSystemGetEnvironment(FTH_FICL_SYSTEM());
	ficlDictionaryAppendConstant(env, "max-n",
	    (ficlInteger) fth_make_llong(LONG_MAX));
	ficlDictionaryAppendConstant(env, "max-u",
	    (ficlInteger) fth_make_ullong(ULONG_MAX));
	ficlDictionaryAppendConstant(env, "max-d",
	    (ficlInteger) fth_make_llong(LLONG_MAX));
	ficlDictionaryAppendConstant(env, "max-ud",
	    (ficlInteger) fth_make_ullong(ULLONG_MAX));
#if !defined(MAXFLOAT)
#define MAXFLOAT		((ficlFloat)3.40282346638528860e+38)
#endif
	ficlDictionaryAppendConstant(env, "max-float",
	    (ficlInteger) fth_make_float(MAXFLOAT));
}

/*
 * numbers.c ends here
 */
