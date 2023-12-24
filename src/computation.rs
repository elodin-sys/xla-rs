use crate::{XlaOp, XlaOpRaw};
use cpp::{cpp, cpp_class};
cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    using namespace xla;
}}

cpp_class!(pub unsafe struct XlaComputation as "XlaComputation");
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
