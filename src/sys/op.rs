use cpp::{cpp, cpp_class};

use super::ArrayShape;
use super::{XlaBuilder, XlaComputation};
use crate::Result;
use crate::{PrimitiveType, Status};
use std::ops::{Add, Div, Mul, Sub};
use std::pin::Pin;

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    using namespace xla;
}}
cpp_class!(pub unsafe struct XlaOp as "XlaOp");

impl XlaOp {
    pub fn build(&self) -> Result<XlaComputation> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let comp = unsafe {
            cpp!([self as "XlaOp*", out_status as "Status*"] -> XlaComputation as "XlaComputation" {
                auto builder = self->builder();
                auto status = builder->Build(*self, false);
                if (status.ok()) {
                    return std::move(status.value());
                }else{
                    *out_status = Status(status.status());
                    return XlaComputation();
                }
            })
        };
        out_status.to_result()?;
        Ok(comp)
    }

    pub fn add(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Add(*self, *rhs));
                } catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Sub(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Mul(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn div(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Div(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn rem(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Rem(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn neg(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Neg(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn abs(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Abs(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn sqrt(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Sqrt(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Pow(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Dot(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn atan2(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Atan2(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn max(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Max(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn min(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Min(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn or(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Or(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn and(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(And(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn xor(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Xor(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn eq(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Eq(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn ne(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Ne(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn ge(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Ge(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn gt(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Gt(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn le(&self, rhs: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Le(*self, *rhs));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn not(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Not(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn exp(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Exp(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn expm1(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Expm1(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn floor(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Floor(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn ceil(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Ceil(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn round(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Round(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn log(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Log(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn log1p(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Log1p(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn logistic(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Logistic(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn sign(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Sign(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn clz(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Clz(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn cos(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Cos(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn sin(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Sin(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn tanh(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Tanh(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn real(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Real(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn imag(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Imag(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn rsqrt(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Rsqrt(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn cbrt(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Cbrt(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn is_finite(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(IsFinite(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn lower_triangle(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(LowerTriangle(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn upper_triangle(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(UpperTriangle(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    /*pub fn einsum1(&self, config: &str,) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", config as "const char*"] -> XlaOp as "XlaOp" {
                return XlaOp(Einsum(*self, config));
            })
        }
    }

    pub fn einsum2(&self, arg2: &Self, config: &str) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", arg2 as "const XlaOp*", config as "const char*"] -> XlaOp as "XlaOp" {
                return XlaOp(Einsum(*self, arg2, config));
            })
        }
    }*/

    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", min as "const XlaOp*", max as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Clamp(*self, *min, *max));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn copy(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Copy(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn zeros_like(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(ZerosLike(*self));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn zero_like(&self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    const Shape *shape = self->builder()->GetShapePtr(*self).value();
                    return XlaOp(Zero(self->builder(), shape->element_type()));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn reshape(&self, ds: &[i64]) -> Self {
        let ds_ptr = ds.as_ptr();
        let ds_len = ds.len();
        unsafe {
            cpp!([self as "const XlaOp*", ds_ptr as "const int64_t*", ds_len as "size_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Reshape(*self, absl::Span(ds_ptr, ds_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn broadcast(&self, ds: &[i64]) -> Self {
        let ds_ptr = ds.as_ptr();
        let ds_len = ds.len();
        unsafe {
            cpp!([self as "const XlaOp*", ds_ptr as "const int64_t*", ds_len as "size_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Broadcast(*self, absl::Span(ds_ptr, ds_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn broadcast_in_dim(&self, dims: &[i64], broadcast_dims: &[i64]) -> Self {
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let broadcast_dims_ptr = broadcast_dims.as_ptr();
        let broadcast_dims_len = broadcast_dims.len();
        unsafe {
            cpp!([self as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t", broadcast_dims_ptr as "const int64_t*", broadcast_dims_len as "int64_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(BroadcastInDim(*self, absl::Span(dims_ptr, dims_len), absl::Span(broadcast_dims_ptr, broadcast_dims_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn collapse(&self, ds: &[i64]) -> Self {
        let ds_ptr = ds.as_ptr();
        let ds_len = ds.len();
        unsafe {
            cpp!([self as "const XlaOp*", ds_ptr as "const int64_t*", ds_len as "size_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Collapse(*self, absl::Span(ds_ptr, ds_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn transpose(&self, dims: &[i64]) -> Self {
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        unsafe {
            cpp!([self as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Transpose(*self, absl::Span(dims_ptr, dims_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn select(&self, on_true: &Self, on_false: &Self) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", on_true as "const XlaOp*", on_false as "const XlaOp*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Select(*self, *on_true, *on_false));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn rng_uniform(&self, sigma: &Self, shape: &ArrayShape) -> Self {
        let dims = shape.dims();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = shape.primitive_type() as i32;
        unsafe {
            cpp!([self as "const XlaOp*", sigma as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> XlaOp as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(RngUniform(*self, *sigma, shape));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn rng_normal(&self, sigma: &Self, shape: &ArrayShape) -> Self {
        let dims = shape.dims();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = shape.primitive_type() as i32;
        unsafe {
            cpp!([self as "const XlaOp*", sigma as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> XlaOp as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(RngNormal(*self, *sigma, shape));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn slice(&self, start_indices: &[i64], limit_indices: &[i64], strides: &[i64]) -> Self {
        let start_indices_ptr = start_indices.as_ptr();
        let start_indices_len = start_indices.len();
        let limit_indices_ptr = limit_indices.as_ptr();
        let limit_indices_len = limit_indices.len();
        let strides_ptr = strides.as_ptr();
        let strides_len = strides.len();
        unsafe {
            cpp!([self as "const XlaOp*", start_indices_ptr as "const int64_t*", start_indices_len as "size_t", limit_indices_ptr as "const int64_t*", limit_indices_len as "size_t", strides_ptr as "const int64_t*", strides_len as "size_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Slice(*self, absl::Span(start_indices_ptr, start_indices_len), absl::Span(limit_indices_ptr, limit_indices_len), absl::Span(strides_ptr, strides_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn slice_in_dim(&self, start_index: i64, limit_index: i64, stride: i64, dim: i64) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", start_index as "int64_t", limit_index as "int64_t", dim as "int64_t", stride as "int64_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(SliceInDim(*self, start_index, limit_index, stride, dim));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn get_tuple_element(&self, index: i64) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", index as "int64_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(GetTupleElement(*self, index));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn gather(
        &self,
        rhs: &Self,
        offset_dims: &[isize],
        slice_dims: &[isize],
        start_index_map: &[isize],
        slice_sizes: &[isize],
    ) -> Self {
        let offset_dims_ptr = offset_dims.as_ptr();
        let offset_dims_len = offset_dims.len();
        let slice_dims_ptr = slice_dims.as_ptr();
        let slice_dims_len = slice_dims.len();
        let start_index_map_ptr = start_index_map.as_ptr();
        let start_index_map_len = start_index_map.len();
        let slice_sizes_ptr = slice_sizes.as_ptr();
        let slice_sizes_len = slice_sizes.len();
        unsafe {
            cpp!([self as "const XlaOp*", rhs as "const XlaOp*", offset_dims_ptr as "const int64_t*", offset_dims_len as "size_t", slice_dims_ptr as "const int64_t*", slice_dims_len as "size_t", start_index_map_ptr as "const int64_t*", start_index_map_len as "size_t", slice_sizes_ptr as "const int64_t*", slice_sizes_len as "size_t"] -> XlaOp as "XlaOp" {
                    GatherDimensionNumbers dn;
                    for (size_t i = 0; i < offset_dims_len; ++i) {
                        dn.add_offset_dims(offset_dims_ptr[i]);
                    }
                    for (size_t i = 0; i < slice_dims_len; ++i) {
                        dn.add_collapsed_slice_dims(slice_dims_ptr[i]);
                    }
                    for (size_t i = 0; i < start_index_map_len; ++i) {
                        dn.add_start_index_map(start_index_map_ptr[i]);
                    }
                    auto ss = absl::Span<const int64_t>(slice_sizes_ptr, slice_sizes_len);
                    return XlaOp(Gather(*self, *rhs, dn, ss));
            })
        }
    }

    pub fn convert_element_type(&self, ty: PrimitiveType) -> Self {
        let ty = ty as i32;
        unsafe {
            cpp!([self as "const XlaOp*", ty as "int32_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(ConvertElementType(*self, (PrimitiveType)ty));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn get_dimension_size(&self, dim: i64) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", dim as "int64_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(GetDimensionSize(*self, dim));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn reduce(&self, init_value: &Self, comp: &XlaComputation, dims: &[i64]) -> Self {
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        unsafe {
            cpp!([self as "const XlaOp*", init_value as "const XlaOp*", comp as "const XlaComputation*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Reduce(*self, *init_value, *comp, absl::Span(dims_ptr, dims_len)));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn conditional(
        &self,
        true_op: &Self,
        on_true: &XlaComputation,
        false_op: &Self,
        on_false: &XlaComputation,
    ) -> Self {
        unsafe {
            cpp!([self as "const XlaOp*", true_op as "const XlaOp*", on_true as "const XlaComputation*", false_op as "const XlaOp*", on_false as "const XlaComputation*"] -> XlaOp as "XlaOp" {
                try {
                    return XlaOp(Conditional(*self, *true_op, *on_true, *false_op, *on_false));
                }catch(std::exception e) {
                    return XlaOp(self->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        }
    }

    pub fn builder(&self) -> &XlaBuilder {
        unsafe {
            cpp!([self as "const XlaOp*"] -> &XlaBuilder as "const XlaBuilder*" {
                return self->builder();
            })
        }
    }
}

macro_rules! bin_op_impl {
    ($trait:ident, $op:tt) => {
        impl $trait for XlaOp {
            type Output = XlaOp;
            fn $op(self, rhs: Self) -> Self {
                XlaOp::$op(&self, &rhs)
            }
        }

        impl<'a> $trait<&'a Self> for &'a XlaOp {
            type Output = XlaOp;
            fn $op(self, rhs: &'a Self) -> XlaOp {
                XlaOp::$op(self, rhs)
            }
        }
    };
}

bin_op_impl!(Add, add);
bin_op_impl!(Sub, sub);
bin_op_impl!(Mul, mul);
bin_op_impl!(Div, div);
