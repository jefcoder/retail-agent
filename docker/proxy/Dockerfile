FROM nginx:alpine

RUN apk add --no-cache gettext nginx-module-njs

COPY docker/proxy/nginx.conf.template /tmp/nginx.conf.template
COPY docker/proxy/validate_model.js /etc/nginx/validate_model.js
COPY docker/proxy/model_pairs.json /etc/nginx/model_pairs.json

CMD ["/bin/sh", "-c", ": ${OPENROUTER_HOST:=openrouter.ai} && export OPENROUTER_HOST && envsubst '${SEARCH_SERVER_URL} ${SEARCH_SERVER_PORT} ${OPENROUTER_HOST}' < /tmp/nginx.conf.template > /etc/nginx/nginx.conf && exec nginx -g 'daemon off;'"]
