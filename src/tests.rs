use super::*;

#[test]
fn test_new_builder() {
    XlaBuilder::new("test");
}

#[test]
fn test_cpu_client() {
    PjRtClient::cpu().expect("client create failed");
}

#[test]
fn test_compile() {
    let client = PjRtClient::cpu().expect("client create failed");
    let builder = XlaBuilder::new("test");
    let a = builder
        .parameter(
            0,
            &ArrayShape::new_with_type(crate::ElementType::F32, vec![]),
            "a",
        )
        .unwrap();
    let b = builder
        .parameter(
            1,
            &ArrayShape::new_with_type(crate::ElementType::F32, vec![]),
            "b",
        )
        .unwrap();
    let add = a.add(&b);
    let comp = builder.build(&add).unwrap();
    client.compile(&comp).unwrap();
}

#[test]
fn test_exec() {
    let client = PjRtClient::cpu().expect("client create failed");
    let builder = XlaBuilder::new("test");
    let a = builder
        .parameter(
            0,
            &ArrayShape::new_with_type(crate::ElementType::F32, vec![]),
            "a",
        )
        .unwrap();
    let b = builder
        .parameter(
            1,
            &ArrayShape::new_with_type(crate::ElementType::F32, vec![]),
            "b",
        )
        .unwrap();
    let add = a.add(&b);
    let comp = builder.build(&add).unwrap();
    let exec = client.compile(&comp).unwrap();
    let mut args = BufferArgs::default();
    let a = client.copy_host_buffer(&[1.0f32], &[]).unwrap();
    let b = client.copy_host_buffer(&[2.0f32], &[]).unwrap();
    args.push(&a);
    args.push(&b);
    let mut res = exec.execute_buffers(args).unwrap();
    let out = res.pop().unwrap();
    let lit = out.to_literal_sync().unwrap();
    assert_eq!(lit.typed_buf::<f32>().unwrap(), &[3.0f32]);
    // let slice = lit.raw_buf();
    // let out = f32::from_le_bytes(slice.try_into().unwrap());
    // assert_eq!(out, 3.0);
}
