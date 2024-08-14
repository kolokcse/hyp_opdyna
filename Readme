Hipergráfok

\item motiváció: vélemények terjedése, ingerületátviteli folyamatok az agyban 

\item a páronkénti kölcsönhatások sok esetben nem elegendők a folyamat megértéséhez 

\item konszenzus keresése: a folyamatot futtatva valamelyik vélemény kerül-e egyértelmű többségbe, vagy egyensúlyi állapot alakul ki

\item két réteg: "háztartás", és "munkahely"; minden csúcs pontosan egy háztartásban, és pontosan egy munkahelyen van; a háztartások csak egy rögzített beosztás, ami sosem változik;  a munkahely az elején sorsolva van, és utána változhat az átlépésekkel 

\item a "munkahely" kezdetben egy véletlen partíció (később lehetne preferential attachment, ahol nem mindenki egy munkahelyen van)

\item kétféle vélemény van, először minden hiperélben öten vannak, és minden csúcs két hiperélben van (legegyszerűbb: klikkes modell megfelelője, a második réteg is egyenletesen választott csoportokból áll)

\item véletlenszerűen, egyenletesen kiválasztunk egy $v$ csúcsot 

\item megszámoljuk, hogy hány másféle vélemény van a csoportjaiban;  ebből jön két arány, $h$, $m$, $(h+0.5\cdot m)/1.5$-nek lineáris függvénye legyen a megváltoztatási valószínűség: 0-ban legyen 0, ha egyedül van, akkor $p$ valószínűséggel változtat; 
(ennek még nézzünk utána)  

\item ettől függ annak valószínűsége, hogy megváltoztatja a véleményét $v$

\item ha nem változtatja meg és kisebbségben maradt, azaz $m'<1/2$, akkor $q\cdot (1/2-m')^+$ valószínűséggel átlép egy másik "munkahely" hiperélbe,  azok közül, ahol többségben van az ő véleménye, egyenletesen választunk azok közül, ahol az ő véleménye többségben van (később: az aránytól, mérettől függhet)


Statisztikák

\item hányan vannak az egyik véleményen 

\item hány olyan háztartás, illetve munkahely van, ahol az egyik vélemény van többségben; a háztartásoknál a pontos eloszlás is érdekes (vagyis: a háztartások hányadrésze 0-5, 1-4, 2-3, stb.); munkahelynél esetleg hisztogramot lehet készíteni az arányokból (ezeknek más a mérete, ezért többféle arány előjön)

\item a munkahelyek méretének eloszlása

\item mérettel súlyozott változata a vélemények arányának




Paraméterbecslés dinamikusan változó hipergráfokon

Tegyük fel, hogy a hiperélek bizonyos valószínűséggel felbomlanak, a csúcsok átkerülhetnek egy másik hiperélbe, és a felbomlás valószínűsége akkor a legnagyobb, ha a csúcs egyedül marad a véleményével a csoportjában. 

KÉRDÉS:
hogyan tudjuk megbecsülni terjedési rátát, illetve a csoportok felbomlásának valószínűségét, mint ismeretlen paramétereket, az összesített statisztikákból, vagyis az adott véleménnyel rendelkező csúcsok számából, és esetleg egyéb, könnyen megfigyelhető statisztikákból?



Kutatasi kérdések:
meddig futtassuk? - nem talál még stabil pontot 50 ezer után sem


ADATOK leírása:

Az adatok csv fájlokban vannak tárolva, elnevezésükben található a beta (vélemény terjedési ráta) és a q (élfelbomlás ráta) érték.

A .csv fájlokban a következő idősorok találhatók:

Minden sor egy 'sample id'-val kezdődik, ez a szimulációhoztartozó id, azonos szimulációbóltartozóknak ez megegyezik, ezután jön a statisztika megnevezése: 

'sum_opinion_A'- az A véleményen lévők száma
'binned_opinions_home' - az A véleményen lévők szerint hány olyan háztartás van, ahol a háztartásoknál a pontos eloszlás 0,1,2,3,4 vagy 5;
'binned_opinions_wp' - az A véleményen lévők szerint hány olyan munkahely van, ahol a háztartásoknál a pontos eloszlás 0,1,2,3,4,5, ... 15;
'binned_edge_sizes_wp' - a munkahelyek az eloszlása méret szerint (listák ahol pl [0,1,2,1,0] azt jelentené hogy a 0 méretűből 0 van az 1 ből 1 a 2 ből 2 stb...) 