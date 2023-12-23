use std::{marker::PhantomData, pin::Pin};

use cpp::{cpp, cpp_class};
use cxx::{let_cxx_string, CxxString, CxxVector, UniquePtr};

use crate::{ArrayElement, ArrayShape, Error, NativeType, PrimitiveType, Result};

mod op;
mod shape;

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
                std::shared_ptr<XlaBuilder> builder(new XlaBuilder("test"));
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

    pub fn concat_in_dim(&self, others: &[XlaOp], dim: i64) -> XlaOp {
        let others_ptr = others.as_ptr();
        let others_len = others.len();
        unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", others_ptr as "const XlaOp*", others_len as "size_t", dim as "int64_t"] -> XlaOp as "XlaOp" {
                return XlaOp(ConcatInDim(self->get(), absl::Span(others_ptr, others_len), dim));
            })
        }
    }

    pub fn tuple(&self, elems: &[XlaOp]) -> XlaOp {
        let elems_ptr = elems.as_ptr();
        let elems_len = elems.len();
        unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", elems_ptr as "const XlaOp*", elems_len as "size_t"] -> XlaOp as "XlaOp" {
                return XlaOp(Tuple(self->get(), absl::Span(elems_ptr, elems_len)));
            })
        }
    }

    pub fn map(&self, args: &[XlaOp], comp: &XlaComputation, dims: &[i64]) -> XlaOp {
        let args_ptr = args.as_ptr();
        let args_len = args.len();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", args_ptr as "const XlaOp*", args_len as "size_t", comp as "const XlaComputation*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOp as "XlaOp" {
                return XlaOp(Map(self->get(), absl::Span(args_ptr, args_len), *comp, absl::Span(dims_ptr, dims_len)));
            })
        }
    }

    pub fn parameter(&self, num: i64, shape: &ArrayShape, name: &str) -> Result<XlaOp> {
        let_cxx_string!(name = name);
        let dims = shape.dims();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = shape.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let op = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t", num as "int64_t", name as "std::string*"] -> XlaOp as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(Parameter((self->get()), num, shape, *name));
                }catch(std::exception e) {
                    return XlaOp((*self)->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        out_status.to_result()?;
        Ok(op)
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
        unsafe {
            cpp!([self as "const XlaComputation*", body as "const XlaComputation*", init_value as "const XlaOp*"] -> XlaOp as "XlaOp" {
                return XlaOp(While(*self, *body, *init_value));
            })
        }
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

    pub fn to_literal(&self) -> Result<Literal> {
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
