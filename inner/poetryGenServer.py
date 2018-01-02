import grpc
import poetry_pb2_grpc
import poetry_gen
from concurrent import futures
import grpc
import poetry_pb2
import time

class GeneratePoetryServicer(poetry_pb2_grpc.GeneratePoetryServicer):
	"""Provides methods that implement functionality of GeneratePoetry server."""
	def __init__(self):
		self.full_model = poetry_gen.FullModel()

	def GeneratePoetry(self, request, context):
		resp = poetry_pb2.Seed()
		resp.Text = self.full_model.predict(request.Text)
		return resp

def serve():
  _ONE_DAY_IN_SECONDS = 86400
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  poetry_pb2_grpc.add_GeneratePoetryServicer_to_server(
      GeneratePoetryServicer(), server)
  address = '[::]:8080'
  server.add_insecure_port(address)
  print(f'Starting server. Listening on {address}')

  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()