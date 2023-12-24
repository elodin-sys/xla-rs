use crate::{Literal, Result, Status};

use cpp::{cpp, cpp_class};

use std::{marker::PhantomData, pin::Pin};

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    using namespace xla;
}}

cpp_class!(pub unsafe struct PjRtBuffer as "std::unique_ptr<PjRtBuffer>");
cpp_class!(pub unsafe struct BufferArgsInner as "std::unique_ptr<std::vector<PjRtBuffer*>>");

impl PjRtBuffer {
    pub(crate) fn is_null(&self) -> bool {
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

pub struct BufferArgs<'a> {
    phantom_data: PhantomData<&'a ()>,
    pub(crate) buffers: BufferArgsInner,
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

impl<'a> FromIterator<&'a PjRtBuffer> for BufferArgs<'a> {
    fn from_iter<T: IntoIterator<Item = &'a PjRtBuffer>>(iter: T) -> Self {
        let mut args = BufferArgs::default();
        for buf in iter {
            args.push(buf);
        }
        args
    }
}
