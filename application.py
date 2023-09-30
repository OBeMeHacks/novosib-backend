from aiohttp import web
import json

from finance_model import Model

async def predict(request):
    global model
    text = await request.text()
    
    client_id = json.loads(text)['clientId']
    try:
        mean, count = model.predict(client_id)
    except RuntimeError:
        return web.Response(headers={
            "Access-Control-Allow-Origin" : "*",
        }, status=404)
    
    data = {"contributionMoney" : mean, "contributionsNumber" : count}
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

model = Model(
    'data/external_features.frt',
    'data/npo_clnts.csv',
    'data/npo_cntrbtrs.csv',
    'data/correct_target.frt')

if __name__ == '__main__':

    web.run_app(app, port=8050)