from aiohttp import web
import json

async def predict(request):
    data = {"contributionMoney" : 10000, "contributionsNumber" : 10}
    return web.json_response(data, headers={
        "Access-Control-Allow-Origin" : "*",
    }, status=200)

async def options(reuest):
    return web.Response(headers={
        "Access-Control-Allow-Origin" : "*",
        "Access-Control-Allow-Headers" : "*",
    }, status=200)

app = web.Application()
app.add_routes([web.post('/predict', predict),
                web.options('/predict', options)])


if __name__ == '__main__':
    web.run_app(app, port=8050)