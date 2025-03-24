from abc import abstractmethod
from typing import List
from cachesaver.typedefs import Request, Batch, Response, SingleRequestModel, BatchRequestModel

class ModelBasic(SingleRequestModel, BatchRequestModel):
    def __init__(self):
        pass

    @abstractmethod
    async def request(self, request: Request) -> Response:
        pass

    @abstractmethod
    async def batch_request(self, batch: Batch) -> List[Response]:
        pass
