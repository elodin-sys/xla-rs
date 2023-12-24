use std::{marker::PhantomData, pin::Pin};

use crate::{ArrayElement, ElementType, Error, NativeType, PrimitiveType, Result};
use bytemuck::AnyBitPattern;
use cpp::{cpp, cpp_class};
use cxx::{let_cxx_string, CxxString, UniquePtr};

mod op;
mod shape;

use num_traits::FromPrimitive;
pub use op::*;
pub use shape::*;

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    #include "xla/pjrt/pjrt_api.h"
    #include "xla/pjrt/pjrt_c_api_client.h"
    #include "xla/pjrt/pjrt_client.h"
    #include "xla/pjrt/pjrt_stream_executor_client.h"
    #include "xla/pjrt/tfrt_cpu_pjrt_client.h"
    #include "xla/pjrt/gpu/gpu_helpers.h"
    #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
    using namespace xla;
}}
cpp_class!(pub unsafe struct PjRtClient as "std::shared_ptr<PjRtClient>");
cpp_class!(pub unsafe struct PjRtBuffer as "std::unique_ptr<PjRtBuffer>");
cpp_class!(pub unsafe struct BufferArgsInner as "std::unique_ptr<std::vector<PjRtBuffer*>>");
cpp_class!(pub unsafe struct XlaBuilder as "std::shared_ptr<XlaBuilder>");
cpp_class!(pub unsafe struct XlaComputation as "XlaComputation");
cpp_class!(pub unsafe struct Status as "Status");
cpp_class!(pub unsafe struct PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>");
cpp_class!(pub unsafe struct Literal as "std::shared_ptr<Literal>");

impl XlaBuilder {
    pub fn new(name: &str) -> Self {
        let_cxx_string!(name = name);
        unsafe {
            cpp!( [name as "std::string*"] -> XlaBuilder as "std::shared_ptr<XlaBuilder>" {
                std::shared_ptr<XlaBuilder> builder(new XlaBuilder(*name));
                return builder;
            })
        }
    }

    pub fn build(&self, op: &XlaOp) -> Result<XlaComputation> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let comp = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", op as "XlaOp*", out_status as "Status*"] -> XlaComputation as "XlaComputation" {
                auto status = (*self)->Build(*op, false);
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

    pub fn concat_in_dim(&self, others: &[XlaOpRef<'_>], dim: i64) -> XlaOp {
        let others_ptr = others.as_ptr();
        let others_len = others.len();
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", others_ptr as "const XlaOp*", others_len as "size_t", dim as "int64_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConcatInDim(self->get(), absl::Span(others_ptr, others_len), dim));
            })
        };
        XlaOp { raw, builder: self.clone() }
    }

    pub fn tuple(&self, elems: &[XlaOpRef<'_>]) -> XlaOp {
        let elems_ptr = elems.as_ptr();
        let elems_len = elems.len();
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", elems_ptr as "const XlaOp*", elems_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(Tuple(self->get(), absl::Span(elems_ptr, elems_len)));
            })
        };
        XlaOp { raw, builder: self.clone() }
    }

    pub fn map(&self, args: &[XlaOpRef<'_>], comp: &XlaComputation, dims: &[i64]) -> XlaOp {
        let args_ptr = args.as_ptr();
        let args_len = args.len();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", args_ptr as "const XlaOp*", args_len as "size_t", comp as "const XlaComputation*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(Map(self->get(), absl::Span(args_ptr, args_len), *comp, absl::Span(dims_ptr, dims_len)));
            })
        };
        XlaOp { raw, builder: self.clone() }
    }

    pub fn parameter(
        &self,
        num: i64,
        element_ty: ElementType,
        dims: &[i64],
        name: &str,
    ) -> Result<XlaOp> {
        let_cxx_string!(name = name);
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = element_ty.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let op = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t", num as "int64_t", name as "std::string*"] -> XlaOpRaw as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(Parameter((self->get()), num, shape, *name));
                }catch(std::exception e) {
                    return XlaOp((*self)->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        out_status.to_result()?;
        Ok(XlaOp { raw: op, builder: self.clone() })
    }

    /// Create a node with a constant value defined by the specified literal.
    pub fn constant_literal(&self, literal: &Literal) -> Result<XlaOp> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let op = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", literal as "std::shared_ptr<Literal>*"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantLiteral(self->get(), *literal->get()));
            })
        };
        out_status.to_result()?;
        Ok(XlaOp { raw: op, builder: self.clone() })
    }

    pub fn constant<T: NativeType>(&self, val: T) -> XlaOp {
        T::constant_r0(&self, val)
    }

    pub fn setup_alias(&self, param_num: u64, output_index: u64) -> Result<()> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", param_num as "uint64_t", output_index as "uint64_t", out_status as "Status*"] {
                try {
                    (*self)->SetUpAlias({(int64_t) output_index}, (int64_t) param_num, {}, HloInputOutputAliasConfig::AliasKind::kMustAlias);
                }catch(std::exception e) {
                    *out_status = Status(tsl::errors::Internal(e.what()));
                }
            })
        };
        out_status.to_result()
    }
}

impl Status {
    pub fn ok() -> Self {
        unsafe {
            cpp!([] -> Status as "Status" {
                return Status();
            })
        }
    }

    pub fn is_ok(&self) -> bool {
        unsafe {
            cpp!([self as "const Status*"] -> bool as "bool" {
                return self->ok();
            })
        }
    }

    pub fn to_result(&self) -> Result<()> {
        if self.is_ok() {
            Ok(())
        } else {
            let msg = unsafe {
                cpp!([self as "Status*"] -> UniquePtr<CxxString> as "std::unique_ptr<std::string>" {
                    return make_unique<std::string>(std::string(self->message()));
                })
            };
            let msg = msg
                .as_ref()
                .and_then(|msg| msg.to_str().ok())
                .map(|msg| msg.to_string())
                .unwrap_or_default();
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            Err(Error::XlaError { msg, backtrace })
        }
    }
}

impl XlaComputation {
    pub fn stmt_while(&self, body: &XlaComputation, init_value: &XlaOp) -> XlaOp {
        let raw = unsafe {
            cpp!([self as "const XlaComputation*", body as "const XlaComputation*", init_value as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(While(*self, *body, *init_value));
            })
        };
        XlaOp { raw, builder: init_value.builder.clone() }
    }
}

impl PjRtClient {
    pub fn cpu() -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let client = unsafe {
            cpp!([out_status as "Status*"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                auto status = xla::GetTfrtCpuClient(false);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
            })
        };
        out_status.to_result()?;
        if client.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError { msg: "Unexpected null pointer".to_string(), backtrace });
        }
        Ok(client)
    }

    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let client = unsafe {
            cpp!([out_status as "Status*", memory_fraction as "double", preallocate as "bool"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
                auto status = GetStreamExecutorGpuClient(false, allocator, 0, 0);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
            })
        };
        out_status.to_result()?;
        if client.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError { msg: "Unexpected null pointer".to_string(), backtrace });
        }
        Ok(client)
    }

    pub fn copy_host_buffer<T: ArrayElement>(
        &self,
        buf: &[T],
        dims: &[usize],
    ) -> Result<PjRtBuffer> {
        let element_count: usize = dims.iter().product();
        if element_count != buf.len() {
            return Err(Error::WrongElementCount { dims: dims.to_vec(), element_count });
        }
        let buf_ptr = buf.as_ptr();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = T::TY.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, []() {}, device
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError { msg: "Unexpected null pointer".to_string(), backtrace });
        }
        Ok(buffer)
    }

    pub fn compile(&self, comp: &XlaComputation) -> Result<PjRtLoadedExecutable> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let exec = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", comp as "const XlaComputation*", out_status as "Status*"] -> PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>" {
                auto client = *self;
                CompileOptions options;
                auto status = client->Compile(*comp, options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtLoadedExecutable>();
                }
            })
        };
        out_status.to_result()?;
        if exec.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError { msg: "Unexpected null pointer".to_string(), backtrace });
        }
        Ok(exec)
    }

    fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtClient>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }
}

impl PjRtBuffer {
    fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtBuffer>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }

    pub fn copy_to_host(&self, out: &mut [u8], offset: usize) -> Result<()> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let out_len = out.len();
        let out_ptr = out.as_mut_ptr();
        unsafe {
            cpp!([self as "std::unique_ptr<PjRtBuffer>*", out_ptr as "uint8_t*", out_len as "size_t", offset as "size_t", out_status as "Status*"] {
                *out_status = (*self)->CopyRawToHost(out_ptr, offset, out_len).Await();
            });
        }
        out_status.to_result()
    }

    pub fn to_literal_sync(&self) -> Result<Literal> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let lit = unsafe {
            cpp!([self as "std::unique_ptr<PjRtBuffer>*", out_status as "Status*"] -> Literal as "std::shared_ptr<Literal>" {
                auto status = (*self)->ToLiteralSync();
                if (status.ok()) {
                    return std::move(status.value());
                }else{
                    *out_status = Status(status.status());
                    return std::make_shared<Literal>(Literal());
                }
            })
        };
        out_status.to_result()?;
        Ok(lit)
    }
}

impl Literal {
    pub fn raw_buf(&self) -> &[u8] {
        let len: Pin<&mut usize> = std::pin::pin!(0);
        let data = unsafe {
            let data = cpp!([self as "std::unique_ptr<Literal>*", len as "size_t*"] -> *const u8 as "const uint8_t*" {
                *len = (*self)->size_bytes();
                return (const uint8_t*) (*self)->untyped_data();
            });
            std::slice::from_raw_parts(data, *len)
        };
        data
    }

    pub fn primitive_type(&self) -> Result<PrimitiveType> {
        let ty = unsafe {
            cpp!([self as "std::unique_ptr<Literal>*"] -> i32 as "int32_t" {
                return (*self)->shape().element_type();
            })
        };
        match FromPrimitive::from_i32(ty) {
            None => Err(Error::UnexpectedElementType(ty)),
            Some(ty) => Ok(ty),
        }
    }

    pub fn element_count(&self) -> usize {
        unsafe {
            cpp!([self as "std::unique_ptr<Literal>*"] -> usize as "size_t" {
                return (*self)->element_count();
            })
        }
    }

    pub fn typed_buf<T: ArrayElement + AnyBitPattern>(&self) -> Result<&[T]> {
        let ty = self.primitive_type()?.element_type()?;
        if ty != T::TY {
            Err(Error::ElementTypeMismatch { on_device: ty, on_host: T::TY })?
        }
        bytemuck::try_cast_slice(self.raw_buf()).map_err(Error::PodCastError)
    }

    pub fn reshape(&self, dims: &[i64]) -> Result<Literal> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let lit = unsafe {
            cpp!([self as "std::unique_ptr<Literal>*", dims_ptr as "const int64_t*", dims_len as "size_t", out_status as "Status*"] -> Literal as "std::shared_ptr<Literal>" {
                auto status = (*self)->Reshape(absl::Span(dims_ptr, dims_len));
                if (status.ok()) {
                    return std::make_shared<Literal>(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::make_shared<Literal>(Literal());
                }
            })
        };
        out_status.to_result()?;
        Ok(lit)
    }
}

impl PjRtLoadedExecutable {
    fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtLoadedExecutable>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }

    pub fn execute_buffers(&self, buffers: BufferArgs) -> Result<Vec<PjRtBuffer>> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffers = buffers.buffers;
        let mut out = vec![];
        {
            let out_ptr = &mut out;
            unsafe {
                cpp!([self as "const std::shared_ptr<PjRtLoadedExecutable>*", buffers as "std::unique_ptr<std::vector<PjRtBuffer*>>", out_status as "Status*", out_ptr as "void*"] {
                    ExecuteOptions options;
                    options.untuple_result = true;
                    auto status = (*self)->Execute(absl::Span(buffers.get(), 1), options);
                    if (status.ok()) {
                        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> bufs = std::move(status).value();
                        for (auto& replica_bufs : bufs) {
                             for (auto& buf : replica_bufs) {
                                 auto out_buf_ptr = rust!(push_out_buf_loaded_exec [out_ptr : &mut Vec<PjRtBuffer> as "void*"] -> *mut PjRtBuffer as "std::unique_ptr<PjRtBuffer>*" {
                                     out_ptr.push(std::mem::transmute(std::ptr::null::<()>()));
                                     let i = out_ptr.len() - 1;
                                     let ptr = &mut out_ptr[i];
                                     ptr as *mut PjRtBuffer
                                 });
                                 *out_buf_ptr = std::move(buf);
                             }
                        }
                    }else{
                        *out_status = Status(status.status());
                    }
                })
            };
        }
        out_status.to_result()?;
        Ok(out)
    }
}

pub struct BufferArgs<'a> {
    phantom_data: PhantomData<&'a ()>,
    buffers: BufferArgsInner,
}

impl<'a> Default for BufferArgs<'a> {
    fn default() -> Self {
        Self {
            phantom_data: Default::default(),
            buffers: unsafe {
                cpp!([] -> BufferArgsInner as "std::unique_ptr<std::vector<PjRtBuffer*>>" {
                    std::unique_ptr<std::vector<PjRtBuffer*>> vec (new std::vector<PjRtBuffer*> {});
                    return vec;
                })
            },
        }
    }
}

impl<'a> BufferArgs<'a> {
    pub fn push(&mut self, buf: &'a PjRtBuffer) {
        let inner = &mut self.buffers;
        let buf = buf as *const PjRtBuffer;
        unsafe {
            cpp!([inner as "std::unique_ptr<std::vector<PjRtBuffer*>>*", buf as "std::unique_ptr<PjRtBuffer>*"] {
                auto buf_ptr = buf->get();
                (*inner)->push_back(buf_ptr);
            })
        };
    }
}

#[cfg(test)]
mod tests;
