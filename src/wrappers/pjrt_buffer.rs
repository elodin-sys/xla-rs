//! A view on a memory slice hosted on a device.
use super::{ElementType, FromPrimitive, Literal, PjRtDevice, Shape};
use crate::{c_lib, Error, Result};
use std::marker::PhantomData;

/// A buffer represents a view on a memory slice hosted on a device.
pub struct PjRtBuffer<'a> {
    pub(super) buffer: c_lib::pjrt_buffer,
    pub(super) marker: PhantomData<&'a super::PjRtClient>,
}

impl<'a> PjRtBuffer<'a> {
    /// Copy the buffer to a different device.
    pub fn copy_to_device(&self, device: PjRtDevice<'a>) -> Result<PjRtBuffer<'a>> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_buffer_copy_to_device(self.buffer, device.device, &mut buffer) };
        super::handle_status(status)?;
        Ok(Self { buffer, marker: PhantomData })
    }

    /// Copy the buffer back to the host as a literal.
    pub fn to_literal_sync(&self) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_buffer_to_literal_sync(self.buffer, &mut result) };
        super::handle_status(status)?;
        Ok(Literal(result))
    }

    /// Retrieve the shape used by this buffer.
    pub fn on_device_shape(&self) -> Result<Shape> {
        let shape = unsafe { c_lib::pjrt_buffer_on_device_shape(self.buffer) };
        let rank = unsafe { c_lib::shape_dimensions_size(shape) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(shape, i) }).collect();
        let element_type = unsafe { c_lib::shape_element_type(shape) };
        unsafe { c_lib::shape_free(shape) };
        match FromPrimitive::from_i32(element_type) {
            None => Err(Error::UnexpectedElementType(element_type)),
            Some(element_type) => Ok(Shape { element_type, dimensions }),
        }
    }

    /// Copy the data stored in a buffer to host memory in a blocking way.
    pub fn copy_raw_to_host_sync<T: ElementType>(
        &self,
        dst: &mut [T],
        offset: usize,
    ) -> Result<()> {
        let shape = self.on_device_shape()?;
        if shape.element_type != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch {
                on_device: shape.element_type,
                on_host: T::PRIMITIVE_TYPE,
            })?
        }
        if offset + dst.len() > shape.element_count() {
            Err(Error::TargetBufferIsTooLarge { offset, shape, buffer_len: dst.len() })?
        }
        let status = unsafe {
            c_lib::pjrt_buffer_copy_raw_to_host_sync(
                self.buffer,
                dst.as_mut_ptr() as *mut libc::c_void,
                offset,
                dst.len() * T::ELEMENT_SIZE_IN_BYTES,
            )
        };
        super::handle_status(status)?;
        Ok(())
    }
}

impl<'a> Drop for PjRtBuffer<'a> {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_buffer_free(self.buffer) }
    }
}
