# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import poetry_pb2 as poetry__pb2


class GeneratePoetryStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GeneratePoetry = channel.unary_unary(
        '/poetry.GeneratePoetry/GeneratePoetry',
        request_serializer=poetry__pb2.Seed.SerializeToString,
        response_deserializer=poetry__pb2.Poetry.FromString,
        )


class GeneratePoetryServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GeneratePoetry(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_GeneratePoetryServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GeneratePoetry': grpc.unary_unary_rpc_method_handler(
          servicer.GeneratePoetry,
          request_deserializer=poetry__pb2.Seed.FromString,
          response_serializer=poetry__pb2.Poetry.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'poetry.GeneratePoetry', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
